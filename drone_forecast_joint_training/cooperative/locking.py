"""동일 가중치 폴더에 학습 프로세스가 겹쳐 쓰지 않도록 파일 잠금."""

import os


class TrainingLock:
    def __init__(self, path):
        self.handle = open(path, "a+b")
        if self.handle.tell() == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise RuntimeError(
                "같은 폴더에서 학습 실행 중. 기존 실행 종료 후 --resume 필요."
            ) from None

    def close(self):
        self.handle.close()
