# -*- coding: utf-8 -*-
"""
This script demonstrates how to control an Andor SDK3 camera using Python
with the pylablib library.

It will:
1. Connect to the first available Andor SDK3 camera.
2. Set the camera to 16-bit (High Dynamic Range) mode by inspecting
   the 'SimplePreAmpGainControl' attribute's options.
3. Set the exposure time to 20ms.
4. Acquire a single image (a "snapshot").
5. Print image properties and sample pixel values.
6. Explicitly stop/flush/deallocate buffers before closing to avoid AT_ERR_COMM.
"""
import pylablib.devices.Andor as Andor
import time
import numpy as np
import cv2


def save_image(img: np.ndarray, base_name="snapshot"):
    """
    Try to save `img`. Prefer 16-bit PNG if dtype==uint16 and cv2 supports it.
    Fallback is not implemented here (kept minimal because you requested not to
    change the main logic), but this will write a 16-bit PNG when dtype==uint16.
    Returns filename saved (or None on failure).
    """
    try:
        if img.dtype == np.uint16:
            fname = f"{base_name}_uint16.png"
            ok = cv2.imwrite(fname, img)
            if ok:
                return fname
            else:
                print("cv2.imwrite failed to write uint16 PNG.")
        else:
            # For non-uint16 we still try to write it as-is (opencv will handle common types)
            fname = f"{base_name}.png"
            ok = cv2.imwrite(fname, img)
            if ok:
                return fname
    except Exception as ex:
        print("Exception occurred while trying to save image:", ex)
    return None


try:
    print("Connecting to Andor SDK3 camera...")
    # The 'with' statement handles open() and close()
    with Andor.AndorSDK3Camera() as cam:
        print("Successfully connected.")
        
        info = cam.get_device_info()
        print(f"Camera Model: {info.camera_model}, Serial Number: {info.serial_number}")

        # --- Set High Dynamic Range (16-bit) Mode ---



        # 2. Check and set SimplePreAmpGainControl for HDR
        if "SimplePreAmpGainControl" in cam.get_all_attributes():
            # Get the attribute object
            gain_attr = cam.get_attribute("SimplePreAmpGainControl")
            
            # Get its options from the .values property
            options = gain_attr.values
            print(f"\nAvailable 'SimplePreAmpGainControl' options: {options}")
            
            target_gain = None
            if "High dynamic range (16-bit)" in options:
                target_gain = "High dynamic range (16-bit)"
            
            if target_gain:
                cam.set_attribute_value("SimplePreAmpGainControl", target_gain)
                current_gain = cam.get_attribute_value("SimplePreAmpGainControl")
                print(f"Set 'SimplePreAmpGainControl' to: {current_gain}")
            else:
                print("Warning: Could not find '16-bit' or 'High Dynamic Range' gain mode.")
        # --------------------------------------------------
        
        # 1. Check and set BitDepth
        # We get the attribute object first using get_attribute()
        if "BitDepth" in cam.get_all_attributes():
                print(f"'BitDepth' is: {cam.get_attribute_value('BitDepth')}")

        # 3. Set up Exposure Time
        new_exposure = 0.02  # 20 milliseconds
        cam.set_attribute_value("ExposureTime", new_exposure)
        print(f"\nSet 'ExposureTime' to: {new_exposure} s")

        actual_exposure = cam.get_attribute_value("ExposureTime")
        print(f"Actual 'ExposureTime' from camera: {actual_exposure:.5f} s")
        
        time.sleep(0.1) 

        # 4. Acquire a single frame
        # snap() handles start, wait, read, and stop.
        print("\nAcquiring a single frame...")
        image_data = cam.snap()

        print("Acquisition complete.")

        # 5. Process the image data
        print(f"Image acquired successfully with shape (Height x Width): {image_data.shape}")
        print(f"Image data type: {image_data.dtype}")

        if image_data.ndim > 1 and image_data.shape[1] > 20:
            print(f"First 20 pixel values of the first row: {image_data[0, :20]}")
        else:
            print(f"Image data sample: {image_data.flatten()[:20]}")
        # Ensure numpy array

        image_data = np.asarray(image_data)
        shape = image_data.shape
        dtype = image_data.dtype
        bytes_len = image_data.nbytes
        mb = bytes_len / (1024.0 ** 2)
        
        print(f"\nImage acquired successfully with shape (H x W{(' x C' if image_data.ndim==3 else '')}): {shape}")
        print(f"Image data type: {dtype}")
        print(f"Image memory: {bytes_len} bytes ({mb:.3f} MB)")

        # Print first 20 pixels (flatten or first row depending on dims)
        flat = image_data.flatten()
        sample = flat[:20]
        print(f"First {len(sample)} pixel values: {sample}")

        # --- Save image (16-bit if possible; fallback to 8-bit) ---
        saved_fname = save_image(image_data, base_name="snapshot")
        if saved_fname:
            print(f"Saved image to: {saved_fname}")
        else:
            print("Image not saved (cv2 write failed).")

        print("\nPerforming explicit safe stop/flush/deallocate BEFORE exiting the 'with' block...")
        cam.clear_acquisition()
        cam.close()
        print("Safe stop/flush/deallocate complete.")
        
except Exception as e:
    print("An error occurred:", e)