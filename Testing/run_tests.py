from HardwareControl.CameraControl.zylaCamera import ZylaCamera
from HardwareControl.andorCameraApp import AndorCameraApp

# with ZylaCamera() as cam:
#     app = AndorCameraApp(cam)
#     app.set_gain_mode("16-bit (low noise & high well capacity)")
#     app.set_exposure(0.02)
#     app.take_and_save_image("my_sample", software_gain=1.5)

# -----------------------

with ZylaCamera() as cam:
    app = AndorCameraApp(cam)
    app.set_exposure(0.001)
    app.set_roi()
    app.start_live_view()  # Press 'q' to quit