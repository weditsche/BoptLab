import xml.etree.ElementTree as ET
from czifile import CziFile

def extract_czi_specific_metadata(czi_filepath):
    with CziFile(czi_filepath) as czi:
        meta_xml = czi.metadata()  # returns a string with XML

    root = ET.fromstring(meta_xml)

    # Helper function to get text from a single tag
    def get_text(xpath):
        elem = root.find(xpath)
        return elem.text.strip() if elem is not None and elem.text else None

    # Helper function to convert meters to micrometers
    def meters_to_micrometers(m_value):
        return float(m_value) * 1e6

    # Acquisition Date & Time
    acquisition_date = get_text('.//Metadata/Information/Image/AcquisitionDateAndTime')

    # Image Dimensions & Bit Depth
    SizeX = get_text('.//Metadata/Information/Image/SizeX')
    SizeY = get_text('.//Metadata/Information/Image/SizeY')
    SizeZ = get_text('.//Metadata/Information/Image/SizeZ')
    SizeT = get_text('.//Metadata/Information/Image/SizeT')
    SizeC = get_text('.//Metadata/Information/Image/SizeC')
    bit_count = get_text('.//Metadata/Information/Image/ComponentBitCount')

    # Channels
    # Channels might be under Dimensions/Channels/Channel
    channels = root.findall('.//Metadata/Information/Image/Dimensions/Channels/Channel')
    channel_info = []
    for ch in channels:
        # Fluor name
        fluor = ch.find('Fluor')
        fluor_name = fluor.text.strip() if fluor is not None else None

        # Excitation Wavelength
        exc_wl_elem = ch.find('ExcitationWavelength')
        exc_wl = exc_wl_elem.text.strip() if exc_wl_elem is not None else None

        # Detection Wavelength (Ranges)
        det_wl_elem = ch.find('DetectionWavelength/Ranges')
        det_wl = det_wl_elem.text.strip() if det_wl_elem is not None else None

        # Voltage (Detector Gain)
        voltage_elem = ch.find('Voltage')
        voltage = voltage_elem.text.strip() if voltage_elem is not None else None

        # Detector ID: <Detector Id="Detector: Airyscan">
        detector_elem = ch.find('Detector')
        detector_id = detector_elem.get('Id') if detector_elem is not None else None

        # Frame Time
        frame_time_elem = ch.find('FrameTime')
        frame_time = frame_time_elem.text.strip() if frame_time_elem is not None else None

        # Pixel Time
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

    # Objective Info
    objective = root.find('.//Metadata/Information/Instrument/Objectives/Objective')
    objective_name = objective.get('Name') if objective is not None else None

    lens_na = get_text('.//Metadata/Information/Instrument/Objectives/Objective/LensNA')
    immersion_ri = get_text('.//Metadata/Information/Instrument/Objectives/Objective/ImmersionRefractiveIndex')
    immersion = get_text('.//Metadata/Information/Instrument/Objectives/Objective/Immersion')

    # Pixel Dimensions (Scaling)
    # Distances under: ZISRAW/Metadata/Scaling/Items/Distance
    def get_distance(axis):
        dist = root.find(f".//Metadata/Scaling/Items/Distance[@Id='{axis}']")
        if dist is not None:
            val_elem = dist.find('Value')
            unit_elem = dist.find('DefaultUnitFormat')
            if val_elem is not None and val_elem.text and unit_elem is not None:
                # Convert from meters to µm
                meters = float(val_elem.text.strip())
                return meters_to_micrometers(meters), unit_elem.text.strip()
        return None, 'µm'

    px_x, px_x_unit = get_distance('X')
    px_y, px_y_unit = get_distance('Y')
    # Z might not exist, handle gracefully
    px_z, px_z_unit = get_distance('Z')
    if px_z is None:
        px_z = 1.0  # default if no Z distance
        px_z_unit = 'µm'

    # Print extracted metadata
    print("Acquisition Date & Time:", acquisition_date)
    print("Image Dimensions:")
    print("  SizeX:", SizeX)
    print("  SizeY:", SizeY)
    print("  SizeZ:", SizeZ)
    print("  SizeT:", SizeT)
    print("  SizeC:", SizeC)
    print("Bit Depth:", bit_count)

    print("Channel Information:")
    for i, ch_info in enumerate(channel_info, start=1):
        print(f"  Channel {i}:")
        for k, v in ch_info.items():
            print(f"    {k}: {v}")

    print("Objective Information:")
    print("  Name:", objective_name)
    print("  Lens NA:", lens_na)
    print("  Immersion Refractive Index:", immersion_ri)
    print("  Immersion:", immersion)

    print("Pixel Sizes (in µm):")
    print("  X:", px_x, px_x_unit)
    print("  Y:", px_y, px_y_unit)
    print("  Z:", px_z, px_z_unit)


if __name__ == "__main__":
    # Update the path to your CZI file
    czi_file = "resources/example_images/3C_2D_testCZI.czi"
    extract_czi_specific_metadata(czi_file)
