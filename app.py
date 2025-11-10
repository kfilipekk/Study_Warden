import cv2
import dlib
import numpy as np
import time
import threading
from flask import Flask, render_template, jsonify, Response, request
from picamera2 import Picamera2
## all the hardware stuff we need to control the GPIO pins
from gpiozero import Servo, Buzzer, DigitalOutputDevice
from gpiozero.pins.lgpio import LGPIOFactory 
## scapy lets us sniff network traffic to catch distracting websites
from scapy.all import sniff, DNSQR

## --- Configuration ---
## --- AI & Model Paths ---
DLIB_MODEL_PATH = "shape_predictor_68_face_landmarks.dat"
YOLO_CONFIG_PATH = "yolov4-tiny.cfg"
YOLO_WEIGHTS_PATH = "yolov4-tiny.weights"
COCO_NAMES_PATH = "coco.names"
with open(COCO_NAMES_PATH, "r") as f:
    CLASSES = [line.strip() for line in f.readlines()]
DISTRACTION_CLASSES = ["cell phone", "tv"]

## websites that are definitely not for studying lol
DISTRACTION_DOMAINS = [
    b"youtube.com", b"reddit.com", b"facebook.com", b"twitter.com",
    b"instagram.com", b"netflix.com", b"9gag.com"
]

## these are the thresholds you can tune from the web interface
ai_settings = {
    'SMILE_THRESHOLD': 1.80,
    'OBJ_CONFIDENCE_THRESHOLD': 0.30,
    'FACE_CONFIDENCE_THRESHOLD': 0.5,
    'BREAK_MINS_PER_HOUR': 1.0
}

## --- Global State ---
current_status = "INITIALIZING"
study_time_seconds, break_time_seconds, distraction_time_seconds = 0, 0, 0
alert_message = ""
device_is_armed = False ## tracks whether we should actually punish or just monitor
punishment_thread = None ## prevents us from running multiple punishments at once
distraction_latch_counter = 0
network_distraction_latch_counter = 0

## these are for drawing debug boxes on the video feed so you can see what the AI sees
debug_face_box = None
debug_distraction_boxes = []
debug_phone_confidence = 0.0
debug_smile_ratio = 0.0
debug_face_confidence = 0.0
ai_lock = threading.Lock()

## video frame globals so we can pass frames between threads safely
latest_frame_bytes = None
latest_ai_frame = None
frame_lock = threading.Lock()
picam2 = Picamera2()

## --- Hardware Setup ---
## setting up all the GPIO devices with the correct pins
try:
    factory = LGPIOFactory()
    gpio_devices = {
        'speaker': Buzzer(0, pin_factory=factory),
        'vibration': DigitalOutputDevice(5, pin_factory=factory),
        'alarm': Buzzer(6, pin_factory=factory),
        'servo': Servo(13, pin_factory=factory)
    }
    print("Hardware initialized successfully with new pin definitions.")
except Exception as e:
    print(f"!!! HARDWARE FAILED TO INITIALIZE: {e} !!!")
    print("!!! GPIO functions will not work. Did you run with 'sudo'? !!!")
    gpio_devices = {} ## empty dict so the app doesn't crash even without hardware

## --- Flask Web App ---
app = Flask(__name__, template_folder='.')

@app.route('/')
def index():
    return render_template('index.html', settings=ai_settings)

@app.route('/set_thresholds', methods=['POST'])
def set_thresholds():
    global ai_settings, ai_lock
    try:
        data = request.get_json()
        with ai_lock:
            if 'smile' in data: ai_settings['SMILE_THRESHOLD'] = float(data['smile'])
            if 'phone' in data: ai_settings['OBJ_CONFIDENCE_THRESHOLD'] = float(data['phone'])
            if 'face' in data: ai_settings['FACE_CONFIDENCE_THRESHOLD'] = float(data['face'])
            if 'break_ratio' in data: ai_settings['BREAK_MINS_PER_HOUR'] = float(data['break_ratio'])
        print(f"--- SETTINGS UPDATED: {ai_settings} ---")
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/trigger_distraction', methods=['POST'])
def trigger_distraction():
    global distraction_latch_counter, ai_lock
    with ai_lock:
        distraction_latch_counter = 5 
    print("--- MANUAL DISTRACTION TRIGGERED ---")
    return jsonify({"status": "success"}), 200

## this endpoint arms the device so it actually starts punishing you
@app.route('/arm_device', methods=['POST'])
def arm_device():
    global device_is_armed, study_time_seconds, break_time_seconds, distraction_time_seconds, ai_lock
    with ai_lock:
        device_is_armed = True
        ## reset all the timers to start fresh
        study_time_seconds = 0
        break_time_seconds = 0
        distraction_time_seconds = 0
    print("--- DEVICE HAS BEEN ARMED. TIMERS RESET. ---")
    return jsonify({"status": "success", "message": "Device armed."}), 200

@app.route('/status')
def get_status():
    with ai_lock:
        status, s_time, b_time, d_time, alert = current_status, study_time_seconds, break_time_seconds, distraction_time_seconds, alert_message
    def format_time(s): return f"{int(s//3600):02}:{int((s%3600)//60):02}:{int(s%60):02}"
    return jsonify({"status": status, "studyTime": format_time(s_time), "breakTime": format_time(b_time), "distractionTime": format_time(d_time), "alert": alert})


@app.route('/test_gpio', methods=['POST'])
def test_gpio():
    device_name = request.get_json().get('device')
    if device_name not in gpio_devices:
        print(f"Error: Invalid device name '{device_name}'")
        return jsonify({"status": "error", "message": f"Invalid device: {device_name}"}), 400
    threading.Thread(target=run_gpio_test, args=(device_name,), daemon=True).start()
    return jsonify({"status": "success", "message": f"Testing {device_name}."}), 200

def run_gpio_test(device_name):
    """Quick hardware test to make sure everything's wired up correctly."""
    print(f"--- Testing hardware device: {device_name}... ---")
    device = gpio_devices.get(device_name)
    if not device: return

    try:
        if device_name == 'speaker':
            device.beep(on_time=0.1, off_time=0.1, n=2)
        elif device_name == 'alarm':
            device.beep(on_time=0.5, off_time=0.5, n=3)
        elif device_name == 'vibration':
            device.on(); time.sleep(3); device.off()
        elif device_name == 'servo':
            ## test the servo with clear movements
            ## gpiozero Servo uses -1 (min) to 1 (max), 0 is middle
            device.min()  ## move to minimum position (-1)
            time.sleep(1)  ## wait for servo to reach position
            device.max()  ## move to maximum position (1)
            time.sleep(1)
            device.mid()  ## return to center (0)
            time.sleep(1)
            device.detach()  ## detach to stop holding position
        print(f"--- Test for '{device_name}' complete. ---")
    except Exception as e:
        print(f"--- Error during '{device_name}' test: {e} ---")

## the full punishment sequence that runs when you've been distracted too long
def run_punishment_sequence():
    """This is the main punishment - it runs in its own thread so it doesn't block everything."""
    
    ## step 1: speaker blasts for 10 seconds straight
    try:
        if 'speaker' in gpio_devices:
            print("Punishment: Speaker ON (10s)")
            gpio_devices['speaker'].on()
            time.sleep(10)
            gpio_devices['speaker'].off()
    except Exception as e:
        print(f"Punishment Error (Speaker): {e}")

    ## step 2: rapid fire alarm beeps and vibration for 15 seconds (this is pretty annoying)
    try:
        if 'alarm' in gpio_devices and 'vibration' in gpio_devices:
            print("Punishment: Alarm/Vibration sequence (15s)")
            start_time = time.time()
            while time.time() - start_time < 15:
                gpio_devices['alarm'].on()
                time.sleep(0.010) ## quick 10ms beep (increased from 5ms)
                gpio_devices['alarm'].off()
                gpio_devices['vibration'].on()
                time.sleep(0.100) ## longer 100ms buzz (increased from 50ms)
                gpio_devices['vibration'].off()
    except Exception as e:
        print(f"Punishment Error (Alarm/Vibe): {e}")
    finally:
        ## make sure everything is off before moving on
        if 'alarm' in gpio_devices: gpio_devices['alarm'].off()
        if 'vibration' in gpio_devices: gpio_devices['vibration'].off()

    ## step 3: servo does its thing (probably sprays water or something equally unpleasant)
    ## oscillates between min and max positions for 10 seconds
    try:
        if 'servo' in gpio_devices:
            print("Punishment: Servo oscillation (10s)")
            start_time = time.time()
            toggle = True
            while time.time() - start_time < 10:
                if toggle:
                    gpio_devices['servo'].min()  ## move to minimum position (-1)
                else:
                    gpio_devices['servo'].max()  ## move to maximum position (1)
                toggle = not toggle
                time.sleep(0.5)  ## give servo time to move (500ms)
            gpio_devices['servo'].mid()  ## return to center
            time.sleep(0.3)
            gpio_devices['servo'].detach()  ## turn off servo after done
    except Exception as e:
        print(f"Punishment Error (Servo): {e}")

    print("--- PUNISHMENT SEQUENCE COMPLETE ---")
    ## device stays armed after punishment so it can happen again if needed

def gen_frames():
    global latest_frame_bytes, frame_lock
    while True:
        with frame_lock:
            if latest_frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + latest_frame_bytes + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def calculate_smile_ratio(shape):
    try:
        p48, p54 = shape.part(48), shape.part(54); mouth_width = np.linalg.norm([p48.x - p54.x, p48.y - p54.y])
        p31, p35 = shape.part(31), shape.part(35); nose_width = np.linalg.norm([p31.x - p35.x, p31.y - p35.y])
        return 0.0 if nose_width == 0 else mouth_width / nose_width
    except: return 0.0

def packet_callback(packet):
    global network_distraction_latch_counter, ai_lock
    if packet.haslayer(DNSQR):
        query = packet[DNSQR].qname
        if any(domain in query for domain in DISTRACTION_DOMAINS):
            print(f"--- NETWORK DISTRACTION DETECTED: {query.decode()} ---")
            with ai_lock: network_distraction_latch_counter = 5

def network_sniffing_thread():
    print("Network Thread: Starting DNS sniffer (requires sudo)...")
    try:
        sniff(filter="udp port 53", prn=packet_callback, store=0)
    except Exception as e:
        print(f"--- NETWORK THREAD ERROR: {e}. Did you run with 'sudo'? ---")

def ai_processing_thread():
    global current_status, study_time_seconds, break_time_seconds, distraction_time_seconds, alert_message
    global latest_ai_frame, frame_lock, ai_settings, distraction_latch_counter, network_distraction_latch_counter
    global debug_face_box, debug_distraction_boxes, debug_phone_confidence, debug_smile_ratio, debug_face_confidence, ai_lock
    global device_is_armed, punishment_thread

    print("AI Thread: Loading AI models...")
    try:
        detector_face = dlib.get_frontal_face_detector()
        predictor_expression = dlib.shape_predictor(DLIB_MODEL_PATH)
        detector_object = cv2.dnn.readNetFromDarknet(YOLO_CONFIG_PATH, YOLO_WEIGHTS_PATH)
        output_layer_names = [detector_object.getLayerNames()[i - 1] for i in detector_object.getUnconnectedOutLayers()]
    except Exception as e:
        print(f"FATAL: Could not load AI models. {e}"); current_status = "ERROR"; return

    print("AI Thread: Models loaded, starting main loop.")
    hardware_init()
    last_status, status_start_time, face_lost_counter = "", time.time(), 0

    while True:
        with frame_lock:
            if latest_ai_frame is None: time.sleep(0.5); continue
            frame_to_process = latest_ai_frame.copy()
        
        with ai_lock: settings = ai_settings.copy()
        (h, w) = frame_to_process.shape[:2]
        gray = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5) ## downscale for faster face detection
        faces, scores, _ = detector_face.run(gray_small, 1)
        face_found = len(faces) > 0 and scores[0] > settings['FACE_CONFIDENCE_THRESHOLD']

        local_face_box, local_dist_boxes, local_phone_conf, local_smile_ratio, local_face_conf = None, [], 0.0, 0.0, 0.0
        is_over_break = False; is_net_distracted = False

        if face_found:
            face_lost_counter = 0
            best_face, best_score = faces[0], scores[0]
            local_face_conf = best_score
            local_face_box = (best_face.left() * 2, best_face.top() * 2, best_face.right() * 2, best_face.bottom() * 2)
            face_box_dlib = dlib.rectangle(*local_face_box)
            
            blob_obj = cv2.dnn.blobFromImage(frame_to_process, 1/255.0, (416, 416), swapRB=True, crop=False)
            detector_object.setInput(blob_obj)
            layerOutputs = detector_object.forward(output_layer_names)
            
            phone_found = False
            for out in layerOutputs:
                for det in out:
                    classID, confidence = np.argmax(det[5:]), det[5:][np.argmax(det[5:])]
                    if CLASSES[classID] == "cell phone": local_phone_conf = max(local_phone_conf, confidence)
                    if confidence > settings['OBJ_CONFIDENCE_THRESHOLD'] and CLASSES[classID] in DISTRACTION_CLASSES:
                        phone_found = True
                        box = det[0:4] * np.array([w, h, w, h]); (cX, cY, wd, ht) = box.astype("int")
                        x, y = int(cX - (wd/2)), int(cY - (ht/2))
                        local_dist_boxes.append((x, y, x + wd, y + ht))

            local_smile_ratio = calculate_smile_ratio(predictor_expression(gray, face_box_dlib))
            is_smiling = local_smile_ratio > settings['SMILE_THRESHOLD']
            
            with ai_lock:
                is_over_break = study_time_seconds > 60 and break_time_seconds > (study_time_seconds * (settings['BREAK_MINS_PER_HOUR'] / 60.0))
                is_net_distracted = network_distraction_latch_counter > 0
                if is_net_distracted: network_distraction_latch_counter -= 1
            is_distracted = phone_found or is_smiling or is_over_break or is_net_distracted
            
            with ai_lock:
                if is_distracted: distraction_latch_counter = 5 ## latch keeps status stable for a few seconds
                else: distraction_latch_counter = max(0, distraction_latch_counter - 1)
                new_status = "DISTRACTED" if distraction_latch_counter > 0 else "STUDYING"
        else:
            face_lost_counter += 1
            new_status = "BREAK" if face_lost_counter > 3 else current_status

        with ai_lock:
            debug_face_box, debug_distraction_boxes, debug_phone_confidence, debug_smile_ratio, debug_face_confidence = \
                local_face_box, local_dist_boxes, local_phone_conf, local_smile_ratio, local_face_conf
            
            ## update the timers based on current status
            elapsed = time.time() - status_start_time
            if new_status == last_status:
                if new_status == "STUDYING": study_time_seconds += elapsed
                elif new_status == "BREAK": break_time_seconds += elapsed
                elif new_status == "DISTRACTED": distraction_time_seconds += elapsed
            
            ## when status changes, update the hardware to match
            if new_status != last_status:
                if new_status == "STUDYING": 
                    hardware_set_studying()
                elif new_status == "BREAK": 
                    hardware_set_break()
                elif new_status == "DISTRACTED": 
                    hardware_set_distracted()
            
            current_status, status_start_time, last_status = new_status, time.time(), new_status
            
            ## check if we need to trigger punishment (only if armed and distracted for 5+ seconds)
            if device_is_armed and current_status == "DISTRACTED" and distraction_time_seconds > 5:
                if punishment_thread is None or not punishment_thread.is_alive():
                    print("--- PUNISHMENT SEQUENCE TRIGGERED ---")
                    alert_message = "PUNISHMENT SEQUENCE ACTIVATED!"
                    distraction_time_seconds = 0 ## reset so we don't trigger again immediately
                    punishment_thread = threading.Thread(target=run_punishment_sequence, daemon=True)
                    punishment_thread.start()
            elif is_over_break: 
                alert_message = "Break limit exceeded!"
            elif is_net_distracted: 
                alert_message = "Network distraction detected!"
            elif current_status == "DISTRACTED":
                alert_message = f"Distraction for {int(distraction_time_seconds)}s..."
            else: 
                alert_message = ""
                
        time.sleep(1) ## run the AI loop once per second

def video_capture_thread():
    global latest_frame_bytes, latest_ai_frame, frame_lock, picam2, debug_face_box, debug_distraction_boxes, debug_phone_confidence, debug_smile_ratio, debug_face_confidence, ai_lock
    
    print("Video Thread: Starting video capture.")
    while True:
        request = picam2.capture_request()
        frame_4_channel = request.make_array("main")
        request.release()
        frame_3_channel = cv2.cvtColor(frame_4_channel, cv2.COLOR_RGBA2BGR)
        with ai_lock: ## grab the debug info and draw it on the frame
            if debug_face_box:
                (x1, y1, x2, y2) = debug_face_box
                cv2.rectangle(frame_3_channel, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_3_channel, f"Face: {debug_face_confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            for box in debug_distraction_boxes:
                (x1, y1, x2, y2) = box
                cv2.rectangle(frame_3_channel, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame_3_channel, f"Smile: {debug_smile_ratio:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame_3_channel, f"Phone: {debug_phone_confidence:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', frame_3_channel)
        with frame_lock:
            if ret: latest_frame_bytes = buffer.tobytes()
            latest_ai_frame = frame_3_channel
        time.sleep(0.03) ## about 30 fps

## --- Hardware Control Functions ---

def hardware_init():
    if not gpio_devices: return
    print("Hardware: Initializing to neutral state.")
    if 'speaker' in gpio_devices: gpio_devices['speaker'].off()
    if 'alarm' in gpio_devices: gpio_devices['alarm'].off()
    if 'vibration' in gpio_devices: gpio_devices['vibration'].off()
    if 'servo' in gpio_devices: gpio_devices['servo'].mid()

def hardware_set_studying():
    if not gpio_devices: return
    if 'vibration' in gpio_devices: gpio_devices['vibration'].off()
    if 'servo' in gpio_devices: gpio_devices['servo'].mid()

def hardware_set_distracted():
    """Gets called when distraction is first detected, but we don't do anything until punishment."""
    if not gpio_devices: return
    ## intentionally doing nothing here - punishment handles everything
    pass

def hardware_set_break():
    if not gpio_devices: return
    if 'vibration' in gpio_devices: gpio_devices['vibration'].off()
    if 'servo' in gpio_devices: gpio_devices['servo'].detach() ## detach servo to save power

def hardware_cleanup():
    print("Hardware: Cleaning up and stopping all devices.")
    if gpio_devices:
        for device in gpio_devices.values():
            if isinstance(device, Servo): device.detach()
            else: device.off()

if __name__ == '__main__':
    try:
        print("Configuring Camera...")
        config = picam2.create_preview_configuration(main={"size": (1296, 972), "format": "XRGB8888"})
        picam2.configure(config); picam2.start()
        print("Camera started.")
        
        threading.Thread(target=ai_processing_thread, daemon=True).start()
        threading.Thread(target=video_capture_thread, daemon=True).start()
        threading.Thread(target=network_sniffing_thread, daemon=True).start()
        
        print("Starting Flask web server...")
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        picam2.stop()
        hardware_cleanup()
        print("Program stopped cleanly.")

