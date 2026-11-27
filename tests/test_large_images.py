"""Large local radiographs: header policy, allocation bounds and Qt geometry."""
import os
import struct
import subprocess
import sys
import unittest
import zlib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image
from utils_new.import_export import images
from utils_new.images.cache import LazyImageCache, CachedImageJob
from utils_new.images.processing import ImageStats, segmentation_uint8


def png_header(path, width, height, color=0):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, color, 0, 0, 0))
                     + chunk(b'IDAT', zlib.compress(b'\0')) + chunk(b'IEND', b''))


class LargeImageContracts(unittest.TestCase):
    def test_header_above_default_pillow_error_limit_without_decoding_or_global_leak(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'большой.png'
            png_header(path, 14700, 14700)
            original = Image.MAX_IMAGE_PIXELS
            with self.assertRaises(Image.DecompressionBombError):
                Image.open(path)
            with patch.object(images, 'read_image_with_fallback', side_effect=AssertionError('pixels decoded')):
                self.assertEqual(images.image_size_from_file(path), (14700, 14700))
            self.assertEqual(Image.MAX_IMAGE_PIXELS, original)
            with self.assertRaises(OSError):
                images.open_pillow_image(Path(directory) / 'missing.png')
            self.assertEqual(Image.MAX_IMAGE_PIXELS, original)

    def test_decode_byte_limit_reports_action_before_calling_native_decoder(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'large.png'
            png_header(path, 20000, 20000, color=2)
            with patch.object(images.cv2, 'imread', side_effect=AssertionError('native decoder called')):
                with self.assertRaisesRegex(OSError, 'тайлы'):
                    images.read_image_with_fallback(path)
                self.assertEqual(images.image_size_from_file(path), (20000, 20000))

    def test_uint16_conversion_is_exact_for_entire_range(self):
        source = np.arange(65536, dtype=np.uint16).reshape(256,256)
        expected = (source.astype(np.uint32) // 256).astype(np.uint8)
        np.testing.assert_array_equal(ImageStats.to_uint8(source), expected)
        np.testing.assert_array_equal(segmentation_uint8(source), expected)
        self.assertEqual(source.max(), 65535)

    def test_cache_enforces_bytes_and_oversized_current_does_not_prefetch_neighbours(self):
        calls = []
        def load(path):
            calls.append(path)
            return {'image': np.zeros(path, np.uint8), 'gray': np.zeros(path, np.uint8)}
        cache = LazyImageCache([40,40,200,20], load, max_bytes=200, max_resident=16)
        try:
            cache.get(0);cache.get(1)
            self.assertEqual(cache.resident_bytes, 160)
            cache.get(2)
            self.assertFalse(cache.is_loaded(0));self.assertFalse(cache.is_loaded(1))
            cache.prefetch([3])
            self.assertNotIn(20, calls)
            result = CachedImageJob(cache,3).run()
            self.assertEqual(result['item']['image'].size, 20)
            self.assertLessEqual(cache.resident_bytes, 200)
        finally:
            cache.shutdown()

    def test_failed_decode_diagnostic_does_not_keep_pixel_buffers_alive(self):
        import gc
        import weakref
        references = []
        def load(path):
            pixels = np.zeros((100,100), np.uint16)
            references.append(weakref.ref(pixels))
            raise ValueError('corrupt image')
        cache = LazyImageCache(['broken'], load)
        try:
            self.assertIsNone(cache.get(0))
            gc.collect()
            self.assertIsNone(references[0]())
            self.assertEqual(str(cache.get_error(0)), 'corrupt image')
        finally:
            cache.shutdown()

    @unittest.skipUnless(os.environ.get('NEURO_RUN_GUI_TESTS') == '1', 'opt-in Qt')
    def test_display_pixel_budget_keeps_source_coordinate_system(self):
        code = '''
import numpy as np
from PyQt5.QtWidgets import QApplication
from ui.image_conversion import numpy_to_qpixmap
from ui.image_view import SmartGraphicsViewNew
app=QApplication([])
image=np.zeros((2000,3000,3),np.uint8)
view=SmartGraphicsViewNew();view.set_annotations([('detect',0,.5,.5,.2,.2)],3000,2000)
pixmap=numpy_to_qpixmap(image,max_pixels=600000)
assert pixmap.width()*pixmap.height()<=601000
view.set_pixmap(pixmap)
assert view.img_width==3000 and view.img_height==2000
assert image.shape==(2000,3000,3)
view.close()
'''
        run = subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=30)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    @unittest.skipUnless(os.environ.get('NEURO_RUN_LARGE_IMAGE_TESTS') == '1', '216MP memory integration')
    def test_real_216mp_16bit_source_loads_and_retains_dimensions(self):
        from utils_new.images.session import load_one_image_item
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'большой.png'
            image = Image.new('I;16', (14700,14700), 32768)
            image.save(path)
            image.close()
            self.assertEqual(images.image_size_from_file(path), (14700,14700))
            item = load_one_image_item(path, max_side=1024)
            self.assertEqual(item['source_size'], (14700,14700))
            self.assertEqual(item['working_size'], (1024,1024))
            self.assertEqual(item['image'].dtype, np.uint8)
            self.assertTrue((item['image']==128).all())
