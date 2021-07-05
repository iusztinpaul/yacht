import os


def build_last_checkpoint_path(storage_dir: str) -> str:
    return build_checkpoints_path(storage_dir, 'last_checkpoint.zip')


def build_checkpoints_path(storage_dir: str, file_name: str) -> str:
    checkpoint_dir = os.path.join(storage_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    return os.path.join(checkpoint_dir, file_name)


def build_graphics_path(storage_dir: str, file_name: str) -> str:
    graphics_dir = os.path.join(storage_dir, 'graphics')
    if not os.path.exists(graphics_dir):
        os.mkdir(graphics_dir)

    return os.path.join(graphics_dir, file_name)
