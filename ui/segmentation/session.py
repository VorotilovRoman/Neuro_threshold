"""A single frame API over lazy browsing and ML's eager training storage."""

from utils_new.segmentation.result import ImageFrame


def current_frame(owner):
    paths = owner.image_paths
    if not paths:
        return None
    index = owner.current_index
    if not 0 <= index < len(paths):
        raise ValueError("Текущий индекс изображения вне набора.")
    cache = getattr(owner, "_lazy_cache", None)
    if cache is not None:
        record = cache.get(index)
    else:
        records = getattr(owner, "_eager_records", [])
        record = (
            records[index]
            if index < len(records)
            else {
                "image": owner.display_images[index],
                "gray": owner.gray_images[index],
            }
        )
    if record is None:
        raise ValueError(f"Не удалось прочитать изображение: {paths[index]}")
    image, gray = record["image"], record["gray"]
    h, w = gray.shape[:2]
    source = record.get("source_size", (w, h))
    return ImageFrame(
        str(paths[index]),
        image,
        gray,
        source,
        (w, h),
        record.get("scale", (w / source[0], h / source[1])),
        (getattr(owner, "_data_revision", 0), index, str(paths[index]), id(image)),
    )


def prefetch(owner):
    cache = getattr(owner, "_lazy_cache", None)
    if cache is not None:
        ahead = cache.prefetch_ahead
        index = owner.current_index
        cache.prefetch(range(index + 1, min(index + ahead + 2, len(owner.image_paths))))
        cache.trim_around(index, keep_behind=3, keep_ahead=ahead + 4)
