import os
import numpy as np
from tifffile import TiffFile
from src.core.imaging import ImageData
from czifile import CziFile
import xml.etree.ElementTree as ET

class FileLoader:
    def __init__(self):
        pass

    def load(self, filepath: str) -> ImageData:
        ext = os.path.splitext(filepath.lower())[1]

        if ext in ['.tif', '.tiff']:
            return self._load_tiff(filepath)
        elif ext == '.czi':
            return self._load_czi(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _load_tiff(self, filepath: str) -> ImageData:
        metadata = self._extract_tiff_metadata(filepath)

        with TiffFile(filepath) as tif:
            data = tif.asarray()

            if data.ndim == 2:  # (Y, X)
                data = data[np.newaxis, np.newaxis, ..., np.newaxis]  # (1, 1, Y, X, 1)
            elif data.ndim == 3:  # (Z, Y, X)
                data = data[:, np.newaxis, :, :, np.newaxis]  # (Z, 1, Y, X, 1)

            pixel_size_xyz = (
                metadata["PhysicalSizeX"],
                metadata["PhysicalSizeY"],
                1.0,  # No Z resolution in basic TIFF
            )
            bit_depth = metadata["SignificantBits"]
            channel_names = [f"Channel {i+1}" for i in range(data.shape[1])]

            return ImageData(data, pixel_size_xyz=pixel_size_xyz, bit_depth=bit_depth, channel_names=channel_names)

    def _extract_tiff_metadata(self, filepath: str):
        with TiffFile(filepath) as tif:
            metadata = {}
            page = tif.pages[0]

            x_res = page.tags.get("XResolution")
            y_res = page.tags.get("YResolution")

            metadata["PhysicalSizeX"] = 1000 / (x_res.value[0] / x_res.value[1]) if x_res else 1.0
            metadata["PhysicalSizeXUnit"] = "µm"
            metadata["PhysicalSizeY"] = 1000 / (y_res.value[0] / y_res.value[1]) if y_res else 1.0
            metadata["PhysicalSizeYUnit"] = "µm"

            metadata["SignificantBits"] = (
                page.tags.get("BitsPerSample").value if page.tags.get("BitsPerSample") else 16
            )
            metadata["Type"] = tif.series[0].dtype.name if tif.series else "unknown"
            metadata["Shape"] = tif.series[0].shape if tif.series else (1, 1, 1, 1, 1)

            return metadata

    def _load_czi(self, filepath: str) -> ImageData:
        metadata = self._extract_czi_metadata(filepath)

        with CziFile(filepath) as czi:
            img = czi.asarray()

            Z, C, Y, X, T = metadata["Shape"]
            # Assume data is (T, Z, C, Y, X) and reorder to (Z, C, Y, X, T)
            if img.ndim == 5:
                img = np.transpose(img, (1, 2, 3, 4, 0))
            elif img.ndim == 4:
                # (Z, C, Y, X)
                img = img[..., np.newaxis]  # Add T dimension
            elif img.ndim == 3:
                # (Y, X, C) fallback
                Z = 1
                T = 1
                C = img.shape[2] if img.shape[2] else 1
                Y = img.shape[0]
                X = img.shape[1]
                img = img[np.newaxis, :, :, :, np.newaxis]
                img = np.transpose(img, (0, 3, 1, 2, 4))

            pixel_size_xyz = (
                metadata["PhysicalSizeX"],
                metadata["PhysicalSizeY"],
                metadata["PhysicalSizeZ"]
            )
            bit_depth = int(metadata["BitCount"]) if "BitCount" in metadata else 16
            channel_names = [ch["Fluor"] if ch["Fluor"] else f"Channel_{i+1}" 
                             for i, ch in enumerate(metadata["Channels"])]

            return ImageData(
                img,
                pixel_size_xyz=pixel_size_xyz,
                bit_depth=bit_depth,
                channel_names=channel_names,
                metadata=metadata
            )

    def _extract_czi_metadata(self, filepath: str):
        def meters_to_micrometers(m_value):
            return float(m_value) * 1e6

        with CziFile(filepath) as czi:
            meta_xml = czi.metadata()
        root = ET.fromstring(meta_xml)

        def get_text(xpath):
            elem = root.find(xpath)
            return elem.text.strip() if elem is not None and elem.text else None

        acquisition_date = get_text('.//Metadata/Information/Image/AcquisitionDateAndTime')

        SizeX = get_text('.//Metadata/Information/Image/SizeX')
        SizeY = get_text('.//Metadata/Information/Image/SizeY')
        SizeZ = get_text('.//Metadata/Information/Image/SizeZ')
        SizeT = get_text('.//Metadata/Information/Image/SizeT')
        SizeC = get_text('.//Metadata/Information/Image/SizeC')
        bit_count = get_text('.//Metadata/Information/Image/ComponentBitCount')

        channels = root.findall('.//Metadata/Information/Image/Dimensions/Channels/Channel')
        channel_info = []
        for ch in channels:
            fluor = ch.find('Fluor')
            fluor_name = fluor.text.strip() if fluor is not None else None

            exc_wl_elem = ch.find('ExcitationWavelength')
            exc_wl = exc_wl_elem.text.strip() if exc_wl_elem is not None else None

            det_wl_elem = ch.find('DetectionWavelength/Ranges')
            det_wl = det_wl_elem.text.strip() if det_wl_elem is not None else None

            voltage_elem = ch.find('Voltage')
            voltage = voltage_elem.text.strip() if voltage_elem is not None else None

            detector_elem = ch.find('Detector')
            detector_id = detector_elem.get('Id') if detector_elem is not None else None

            frame_time_elem = ch.find('FrameTime')
            frame_time = frame_time_elem.text.strip() if frame_time_elem is not None else None

            pixel_time_elem = ch.find('PixelTime')
            pixel_time = pixel_time_elem.text.strip() if pixel_time_elem is not None else None

            channel_info.append({
                "Fluor": fluor_name,
                "ExcitationWavelength": exc_wl,
                "DetectionWavelength": det_wl,
                "Voltage": voltage,
                "DetectorID": detector_id,
                "FrameTime": frame_time,
                "PixelTime": pixel_time
            })

        objective = root.find('.//Metadata/Information/Instrument/Objectives/Objective')
        objective_name = objective.get('Name') if objective is not None else None
        lens_na = get_text('.//Metadata/Information/Instrument/Objectives/Objective/LensNA')
        immersion_ri = get_text('.//Metadata/Information/Instrument/Objectives/Objective/ImmersionRefractiveIndex')
        immersion = get_text('.//Metadata/Information/Instrument/Objectives/Objective/Immersion')

        def get_distance(axis):
            dist = root.find(f".//Metadata/Scaling/Items/Distance[@Id='{axis}']")
            if dist is not None:
                val_elem = dist.find('Value')
                unit_elem = dist.find('DefaultUnitFormat')
                if val_elem is not None and val_elem.text and unit_elem is not None:
                    meters = float(val_elem.text.strip())
                    return meters_to_micrometers(meters), unit_elem.text.strip()
            return None, 'µm'

        px_x, px_x_unit = get_distance('X')
        px_y, px_y_unit = get_distance('Y')
        if px_x is None:
            px_x = 1.0
        if px_y is None:
            px_y = 1.0
        px_z, px_z_unit = get_distance('Z')
        if px_z is None:
            px_z = 1.0
            px_z_unit = 'µm'

        Z = int(SizeZ) if SizeZ else 1
        C = int(SizeC) if SizeC else len(channel_info) or 1
        Y = int(SizeY) if SizeY else 1
        X = int(SizeX) if SizeX else 1
        T = int(SizeT) if SizeT else 1

        metadata = {
            "AcquisitionDate": acquisition_date,
            "Shape": (Z, C, Y, X, T),
            "BitCount": bit_count if bit_count else 16,
            "Channels": channel_info,
            "ObjectiveName": objective_name,
            "LensNA": lens_na,
            "ImmersionRI": immersion_ri,
            "Immersion": immersion,
            "PhysicalSizeX": px_x,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": px_y,
            "PhysicalSizeYUnit": "µm",
            "PhysicalSizeZ": px_z,
            "PhysicalSizeZUnit": "µm"
        }

        return metadata
