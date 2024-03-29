from pytorch_lightning import LightningModule
from torch import nn

class FreezeModule(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._freezables = nn.ModuleList()

    def register_freezable(self, name, freezable) -> None:
        setattr(self, name, freezable)
        assert hasattr(freezable, "freeze") and callable(freezable.freeze), \
            f"Freezable {name} must have a freeze method"
        self._freezables.append(freezable)

    def on_train_epoch_end(self) -> None:
        """
        Callback function called at the end of each training epoch.

        This function freezes the freezables if it is the first epoch.

        Returns:
            None
        """
        # Check if it is the first epoch
        if self.current_epoch == 0:
            # Freeze the freezables
            for f in self._freezables:
                f.freeze()

        return super().on_train_epoch_end()
