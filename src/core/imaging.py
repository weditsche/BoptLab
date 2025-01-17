# src/core/imaging.py
import numpy as np

class ImageData:
    def __init__(self, data: np.ndarray, 
                 pixel_size_xyz=(1.0, 1.0, 1.0), 
                 bit_depth=16, 
                 channel_names=None,
                 metadata=None):
        """
        data: np.ndarray expected shape convention: (Z, C, Y, X, T)
        pixel_size_xyz: tuple of (x_size, y_size, z_size) in micrometers
        bit_depth: integer representing the image bit depth (e.g., 8, 16, 32)
        channel_names: list of channel names or None if not available
        metadata: dict containing additional metadata fields extracted from the file
                  If a field isn't present, defaults to None.
        """
        self.data = data
        self.pixel_size_xyz = pixel_size_xyz
        self.bit_depth = bit_depth
        self.channel_names = channel_names if channel_names is not None else []
        
        # Use metadata or an empty dict if None
        self.additional_metadata = metadata if metadata is not None else {}
        
        # Safely retrieve each field with defaults
        self.AcquisitionDate = self.additional_metadata.get("AcquisitionDate", None)
        self.Shape = self.additional_metadata.get("Shape", None)
        self.BitCount = self.additional_metadata.get("BitCount", self.bit_depth)
        self.Channels = self.additional_metadata.get("Channels", None)
        self.ObjectiveName = self.additional_metadata.get("ObjectiveName", None)
        self.LensNA = self.additional_metadata.get("LensNA", None)
        self.ImmersionRI = self.additional_metadata.get("ImmersionRI", None)
        self.Immersion = self.additional_metadata.get("Immersion", None)
        self.PhysicalSizeX = self.additional_metadata.get("PhysicalSizeX", self.pixel_size_xyz[0])
        self.PhysicalSizeXUnit = self.additional_metadata.get("PhysicalSizeXUnit", "µm")
        self.PhysicalSizeY = self.additional_metadata.get("PhysicalSizeY", self.pixel_size_xyz[1])
        self.PhysicalSizeYUnit = self.additional_metadata.get("PhysicalSizeYUnit", "µm")
        self.PhysicalSizeZ = self.additional_metadata.get("PhysicalSizeZ", self.pixel_size_xyz[2])
        self.PhysicalSizeZUnit = self.additional_metadata.get("PhysicalSizeZUnit", "µm")

    @property
    def shape(self):
        return self.data.shape
    
    def get_array(self):
        return self.data

    def get_metadata(self):
        # Merge core attributes with additional_metadata
        base_metadata = {
            "pixel_size_xyz": self.pixel_size_xyz,
            "bit_depth": self.bit_depth,
            "channel_names": self.channel_names
        }
        return {**base_metadata, **self.additional_metadata}
