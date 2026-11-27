"""Masked image pretraining: visible-token ViT MAE and gated dense FCMAE.

FCMAE emulates active-site convolution with dense tensors and explicit gates;
this preserves masking semantics but not sparse-kernel memory/compute savings.
"""
import torch
from torch import nn
from torch.nn import functional as F


def patchify(image, patch):
    n, c, h, w = image.shape
    return image.reshape(n,c,h//patch,patch,w//patch,patch).permute(0,2,4,3,5,1).reshape(n,-1,patch*patch*c)


def unpatchify(patches, patch, height, width, channels):
    return patches.reshape(-1,height//patch,width//patch,patch,patch,channels).permute(0,5,1,3,2,4).reshape(-1,channels,height,width)


def position_embedding(height, width, dimension):
    y,x = torch.meshgrid(torch.arange(height),torch.arange(width),indexing='ij')
    omega = 1 / (10000 ** (torch.arange(dimension//4,dtype=torch.float32)/(dimension//4)))
    values = [axis.flatten()[:,None]*omega[None] for axis in (x,y)]
    return torch.cat([part for value in values for part in (value.sin(),value.cos())],dim=-1)[None]


class MaskedAutoencoder(nn.Module):
    """Shared mask policy and deterministic blind reconstruction at application."""
    def __init__(self,cfg):
        super().__init__()
        self.cfg = dict(cfg)
        self.patch = cfg['patch_size']
        self.height,self.width = cfg['img_h'],cfg['img_w']
        self.channels = cfg['in_channels']
        self.grid = (self.height//self.patch,self.width//self.patch)
        self.tokens = self.grid[0]*self.grid[1]
        self.hidden = max(1,min(self.tokens-1,round(self.tokens*cfg['mask_ratio'])))

    def random_mask(self,inputs,valid=None):
        order = torch.rand(inputs.shape[0],self.tokens,device=inputs.device).argsort(1)
        mask = torch.zeros(inputs.shape[0],self.tokens,device=inputs.device,dtype=torch.bool)
        mask.scatter_(1,order[:,:self.hidden],True)
        if valid is not None:
            useful = patchify(valid,self.patch).sum(-1)>0
            # Extremely thin letterboxed images must still contribute to loss.
            empty = ~(useful & mask).any(1)
            for index in empty.nonzero().flatten():
                candidates = useful[index].nonzero().flatten()
                if not len(candidates):raise ValueError('Нет полезных пикселей для masked loss.')
                mask[index,order[index,0]] = False
                mask[index,candidates[torch.randint(len(candidates),(),device=inputs.device)]] = True
        return mask

    def mask_pixels(self,mask):
        return mask.reshape(-1,1,*self.grid).repeat_interleave(self.patch,2).repeat_interleave(self.patch,3)

    def forward_masked(self,inputs,mask=None,valid=None):
        if tuple(inputs.shape[1:]) != (self.channels,self.height,self.width):
            raise ValueError('MAE/FCMAE: вход не соответствует сохранённым C/H/W.')
        mask = self.random_mask(inputs,valid) if mask is None else mask
        if mask.shape != (len(inputs),self.tokens) or mask.dtype != torch.bool:
            raise ValueError('Неверный размер/тип маски patches.')
        counts = mask.sum(1)
        if not ((counts>0)&(counts<self.tokens)).all() or not (counts==counts[0]).all():
            raise ValueError('Нужны скрытые и видимые patches с одинаковым количеством в batch.')
        prediction = self.reconstruct(inputs,mask)
        return prediction, {'masked_pixels':self.mask_pixels(mask)}

    def validation_mask(self,inputs,valid=None):
        generator = torch.Generator().manual_seed(self.cfg['mask_eval_seed'])
        order = torch.randperm(self.tokens,generator=generator).to(inputs.device)
        mask = torch.zeros(len(inputs),self.tokens,dtype=torch.bool,device=inputs.device)
        mask[:,order[:self.hidden]] = True
        if valid is not None:
            useful = patchify(valid,self.patch).sum(-1)>0
            for index in (~(useful & mask).any(1)).nonzero().flatten():
                candidates = useful[index].nonzero().flatten()
                if not len(candidates):raise ValueError('Нет полезных пикселей для masked validation.')
                mask[index,order[0]] = False
                mask[index,candidates[0]] = True
        return mask

    def forward(self,inputs):
        if self.training:return self.forward_masked(inputs)
        # Every pixel is reconstructed only while its patch is hidden. Never
        # evaluate anomaly error using an unmasked pass / copied visible pixels.
        generator = torch.Generator().manual_seed(self.cfg['mask_eval_seed'])
        order = torch.randperm(self.tokens,generator=generator).to(inputs.device)
        result,counts = torch.zeros_like(inputs),torch.zeros_like(inputs[:,:1])
        for start in range(0,self.tokens,self.hidden):
            indices = order[(torch.arange(self.hidden,device=inputs.device)+start)%self.tokens]
            mask = torch.zeros(len(inputs),self.tokens,dtype=torch.bool,device=inputs.device)
            mask[:,indices] = True
            predicted,extra = self.forward_masked(inputs,mask)
            pixels = extra['masked_pixels']
            result = result + predicted*pixels
            counts = counts + pixels
        return result/counts.clamp_min(1), None

    def encode_features(self,inputs):
        """Full-context NCHW encoder grid for downstream transfer, not anomaly scoring."""
        if tuple(inputs.shape[1:])!=(self.channels,self.height,self.width):
            raise ValueError('Вход encoder не соответствует сохранённым C/H/W.')
        visible = torch.zeros(len(inputs),self.tokens,dtype=torch.bool,device=inputs.device)
        encoded,_ = self.encode_visible(inputs,visible)
        if encoded.ndim==3:
            encoded = encoded[:,1:].transpose(1,2).reshape(len(inputs),-1,*self.grid)
        return encoded


class ViTMAE(MaskedAutoencoder):
    """Asymmetric ViT: encoder sees visible tokens only, decoder restores all."""
    def __init__(self,cfg):
        super().__init__(cfg)
        dim,dec = cfg['encoder_dim'],cfg['decoder_dim']
        self.patch_embed = nn.Conv2d(self.channels,dim,self.patch,stride=self.patch)
        self.cls_token = nn.Parameter(torch.zeros(1,1,dim))
        self.register_buffer('encoder_position',position_embedding(*self.grid,dim))
        self.register_buffer('decoder_position',position_embedding(*self.grid,dec))
        def transformer(width,depth,heads):
            layer = nn.TransformerEncoderLayer(width,heads,width*4,dropout=0,activation='gelu',batch_first=True,norm_first=True)
            return nn.TransformerEncoder(layer,depth,norm=nn.LayerNorm(width,eps=1e-6),enable_nested_tensor=False)
        self.mae_encoder = transformer(dim,cfg['encoder_depth'],cfg['encoder_heads'])
        self.decoder_embed = nn.Linear(dim,dec)
        self.mask_token = nn.Parameter(torch.zeros(1,1,dec))
        self.decoder = transformer(dec,cfg['decoder_depth'],cfg['decoder_heads'])
        self.prediction = nn.Linear(dec,self.patch*self.patch*self.channels)
        nn.init.normal_(self.cls_token,std=.02);nn.init.normal_(self.mask_token,std=.02)
        # TransformerEncoder clones a layer; initialize independent weights.
        for module in self.modules():
            if isinstance(module,nn.Linear):nn.init.xavier_uniform_(module.weight);nn.init.zeros_(module.bias)
            if isinstance(module,nn.MultiheadAttention):nn.init.xavier_uniform_(module.in_proj_weight)

    def encode_visible(self,inputs,mask):
        tokens = self.patch_embed(inputs).flatten(2).transpose(1,2)+self.encoder_position
        visible = (~mask).nonzero()[:,1].reshape(len(inputs),-1)
        tokens = tokens.gather(1,visible[:,:,None].expand(-1,-1,tokens.shape[-1]))
        encoded = self.mae_encoder(torch.cat((self.cls_token.expand(len(inputs),-1,-1),tokens),1))
        return encoded,visible

    def reconstruct(self,inputs,mask):
        encoded,visible = self.encode_visible(inputs,mask)
        encoded = self.decoder_embed(encoded)
        full = self.mask_token.expand(len(inputs),self.tokens,-1).clone()
        full.scatter_(1,visible[:,:,None].expand(-1,-1,full.shape[-1]),encoded[:,1:])
        decoded = self.decoder(torch.cat((encoded[:,:1],full+self.decoder_position),1))
        return unpatchify(self.prediction(decoded[:,1:]),self.patch,self.height,self.width,self.channels)


class FCMAE(MaskedAutoencoder):
    """ConvNeXt V2 encoder, mask-aware dense gates, light convolutional decoder."""
    def __init__(self,cfg):
        super().__init__(cfg)
        from utils_new.autoencoders.models import ConvNeXtV2Block
        widths = [cfg['base']*factor for factor in (1,2,4,8)]
        self.stem = nn.Conv2d(self.channels,widths[0],4,stride=4)
        self.stem_norm = nn.LayerNorm(widths[0],eps=1e-6)
        self.down = nn.ModuleList([nn.Conv2d(a,b,2,stride=2) for a,b in zip(widths,widths[1:])])
        self.down_norm = nn.ModuleList([nn.LayerNorm(c,eps=1e-6) for c in widths[:-1]])
        self.fcmae_stages = nn.ModuleList([nn.ModuleList([ConvNeXtV2Block(c) for _ in range(cfg['blocks_per_stage'])]) for c in widths])
        self.decoder_embed = nn.Conv2d(widths[-1],cfg['decoder_dim'],1)
        self.mask_token = nn.Parameter(torch.zeros(1,cfg['decoder_dim'],1,1))
        self.decoder = nn.Sequential(*[ConvNeXtV2Block(cfg['decoder_dim']) for _ in range(cfg['decoder_depth'])])
        self.prediction = nn.Conv2d(cfg['decoder_dim'],self.patch*self.patch*self.channels,1)
        nn.init.normal_(self.mask_token,std=.02)

    def encode_visible(self,inputs,mask):
        visible = (~mask).reshape(-1,1,*self.grid).to(inputs.dtype)
        h = self.stem(inputs*self.mask_pixels(~mask))
        h = self.stem_norm(h.permute(0,2,3,1)).permute(0,3,1,2)
        for index,stage in enumerate(self.fcmae_stages):
            if index:
                before = F.interpolate(visible,size=h.shape[-2:],mode='nearest')
                h = self.down_norm[index-1](h.permute(0,2,3,1)).permute(0,3,1,2)*before
                h = self.down[index-1](h)
            gate = F.interpolate(visible,size=h.shape[-2:],mode='nearest')
            h = h*gate
            for block in stage:h = block(h,visible=gate)
        return h,visible

    def reconstruct(self,inputs,mask):
        encoded,visible = self.encode_visible(inputs,mask)
        embedded = self.decoder_embed(encoded)
        h = embedded*visible+self.mask_token*(1-visible)
        patches = self.prediction(self.decoder(h)).flatten(2).transpose(1,2)
        return unpatchify(patches,self.patch,self.height,self.width,self.channels)
