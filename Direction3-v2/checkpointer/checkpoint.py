import os
import glob
import torch
from detectron2.checkpoint import DetectionCheckpointer

class DetectionCheckpointer_one(DetectionCheckpointer):
    def save(self, name: str, **kwargs) -> None:
        """
        Dump model and checkpointables to a file.
        Args:
            name (str): name of the file.
            kwargs (dict): extra arbitrary data to save.
        """
        if not self.save_dir or not self.save_to_disk:
            return

        #Delete previous model
        filelist=glob.glob(self.save_dir + "/*.pth")
        for file in filelist:
            print('remove {}'.format(file))
            os.remove(file)         
        #save new model
        data = {}
        data["model"] = self.model.state_dict()
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = "{}.pth".format(name)
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving checkpoint to {}".format(save_file))
        with self.path_manager.open(save_file, "wb") as f:
            torch.save(data, f)
        self.tag_last_checkpoint(basename)
