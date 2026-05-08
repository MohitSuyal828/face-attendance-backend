"""
Smart Attendance System
A Flask-based face recognition attendance system using DeepFace (Facenet model).
Supports webcam capture, photo upload, and a JSON API for mobile clients.
"""

import os
import time

# Disable CUDA before TensorFlow loads to avoid driver-init crashes on CPU-only machines
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import io
import json
import base64
import pickle
import logging
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import pytz
from flask import Flask, render_template, request, jsonify, send_file

try:
    from deepface import DeepFace
except Exception as e:  # pragma: no cover
    DeepFace = None
    print(f"❌ DeepFace not installed properly: {e}")

try:
    from mtcnn import MTCNN
except Exception as e:  # pragma: no cover
    MTCNN = None
    print(f"❌ mtcnn not installed properly: {e}")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

app = Flask(__name__)

STUDENT_PHOTOS_DIR = "student_photos"
ATTENDANCE_DIR = "attendance_records"
ENCODINGS_FILE = "face_encodings.pkl"

# DeepFace model settings
FACE_MODEL = "Facenet"  # Lightweight & fast
DISTANCE_METRIC = "cosine"
RECOGNITION_THRESHOLD = 0.40  # DeepFace's recommended threshold for Facenet + cosine
RECOGNITION_MARGIN = 0.05     # Best match must beat 2nd-best by this much (anti-ambiguity)
MIN_FACE_SIZE = 60            # Skip face crops smaller than this (likely false detections)

# We use the standalone `mtcnn` package directly for detection (much more
# accurate than Haar Cascade) and DeepFace only for the Facenet embedding.
# Going through DeepFace's own MTCNN integration triggered TF graph conflicts
# inside Flask that killed the worker.
FACE_PADDING = 0.15  # Pad the MTCNN bounding box by this fraction before embedding

# Downscale large input images before running MTCNN. Large images (e.g. a
# 4608px DSLR/phone photo) can take several seconds to detect on; resizing to
# ~600px wide cuts that down to <300ms with no measurable accuracy loss for
# faces ≥ MIN_FACE_SIZE in the original. Bounding boxes are scaled back to
# original-image coordinates before being returned, so the API response is
# byte-for-byte identical to before.
PROCESSING_MAX_WIDTH = 600

# Timezone for attendance timestamps (India Standard Time)
IST = pytz.timezone("Asia/Kolkata")

# Simple admin credentials (no DB needed)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "1234"


def _check_admin(data):
    """Return True if the request body contains valid admin credentials."""
    if not isinstance(data, dict):
        return False
    return (
        data.get("username") == ADMIN_USERNAME
        and data.get("password") == ADMIN_PASSWORD
    )


# Always make sure storage folders exist
os.makedirs(STUDENT_PHOTOS_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# Setup logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("attendance")

# Globals
student_embeddings = {}  # {student_name: [embedding, ...]}
face_cascade = None      # Lazy Haar cascade (kept for compatibility, unused now)
face_detector = None     # Lazy MTCNN instance


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------


def init_face_cascade():
    """Initialize Haar Cascade once, then reuse. Kept for backwards-compatibility."""
    global face_cascade
    if face_cascade is None:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return face_cascade


def init_face_detector():
    """Initialize MTCNN once, then reuse."""
    global face_detector
    if face_detector is None and MTCNN is not None:
        face_detector = MTCNN()
    return face_detector


def _resize_for_processing(img_bgr):
    """Downscale large images so MTCNN runs in milliseconds instead of seconds.
    Returns (work_img, scale) where `scale` is the factor needed to map
    coordinates from work_img back to the original (orig = work * scale)."""
    h, w = img_bgr.shape[:2]
    if w <= PROCESSING_MAX_WIDTH:
        return img_bgr, 1.0
    scale = w / float(PROCESSING_MAX_WIDTH)
    new_w = PROCESSING_MAX_WIDTH
    new_h = max(1, int(round(h / scale)))
    work = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return work, scale


def _detect_and_align(img_bgr):
    """Run MTCNN on a BGR image, return list of (aligned_crop, bbox, confidence).
    bbox = (x, y, w, h) in ORIGINAL-image coordinates (so the JSON response
    format never changes, even when we downscale internally for speed).

    Alignment strategy:
      1. Compute the angle between the eye keypoints in *image* coords.
      2. Rotate the FULL image around the midpoint between the eyes.
      3. Crop the (padded) bounding box from the rotated image.
    Doing the rotation around the eye midpoint (not the crop center) keeps the
    face inside the frame and the eyes horizontal — which is exactly what
    Facenet was trained on.
    """
    detector = init_face_detector()
    if detector is None:
        return []

    # Downscale before detection — major speed win on large photos
    img_bgr, scale = _resize_for_processing(img_bgr)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_bgr.shape[:2]
    faces = detector.detect_faces(rgb)

    out = []
    for f in faces:
        x, y, w, h = f["box"]
        x, y = max(0, x), max(0, y)
        w, h = max(1, w), max(1, h)
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            continue
        if f.get("confidence", 0) < 0.90:
            continue

        kp = f.get("keypoints") or {}
        le, re = kp.get("left_eye"), kp.get("right_eye")

        source = img_bgr
        if le and re:
            dx = re[0] - le[0]
            dy = re[1] - le[1]
            # Sanity check: in image coordinates, the "left eye" landmark is
            # the eye that appears on the left side of the photo, so dx should
            # be positive. If it's not, MTCNN got confused — skip alignment.
            if dx > 0:
                angle = float(np.degrees(np.arctan2(dy, dx)))
                # Tiny angles aren't worth the warp overhead
                if abs(angle) > 1.0:
                    eye_mid = ((le[0] + re[0]) / 2.0, (le[1] + re[1]) / 2.0)
                    M = cv2.getRotationMatrix2D(eye_mid, angle, 1.0)
                    source = cv2.warpAffine(
                        img_bgr, M, (W, H),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE,
                    )
                    # Re-detect on the rotated image so the bbox matches the
                    # rotated face precisely (avoids rotation artifacts)
                    rgb_rot = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
                    rfaces = detector.detect_faces(rgb_rot)
                    if rfaces:
                        # Pick the face closest to the original eye midpoint
                        rf = min(
                            rfaces,
                            key=lambda r: (
                                (r["box"][0] + r["box"][2] / 2 - eye_mid[0]) ** 2
                                + (r["box"][1] + r["box"][3] / 2 - eye_mid[1]) ** 2
                            ),
                        )
                        x, y, w, h = rf["box"]
                        x, y = max(0, x), max(0, y)
                        w, h = max(1, w), max(1, h)

        # Pad the bounding box so the embedding sees full face context
        pad_x = int(w * FACE_PADDING)
        pad_y = int(h * FACE_PADDING)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(W, x + w + pad_x)
        y1 = min(H, y + h + pad_y)
        crop = source[y0:y1, x0:x1]

        # Scale the bounding box back to ORIGINAL image coordinates so the
        # JSON response (location field) matches the client's image. The crop
        # itself stays at processing-resolution — Facenet resizes to 160×160
        # internally, so cropping a 600px-source crop is just as accurate as
        # cropping the original and faster.
        if scale != 1.0:
            orig_box = (
                int(round(x * scale)),
                int(round(y * scale)),
                int(round(w * scale)),
                int(round(h * scale)),
            )
        else:
            orig_box = (x, y, w, h)
        out.append((crop, orig_box, float(f.get("confidence", 0))))
    return out


def warmup_deepface():
    """Pre-load Facenet (DeepFace) and MTCNN (standalone) at startup so the
    first request doesn't pay the model-load cost mid-request."""
    if DeepFace is not None:
        try:
            log.info("Warming up Facenet (%s)...", FACE_MODEL)
            dummy = np.zeros((160, 160, 3), dtype=np.uint8)
            DeepFace.represent(
                img_path=dummy,
                model_name=FACE_MODEL,
                enforce_detection=False,
                detector_backend="skip",
            )
            log.info("✅ Facenet ready")
        except Exception as e:
            log.warning("Facenet warmup failed (will retry on first request): %s", e)

    if MTCNN is not None:
        try:
            log.info("Warming up MTCNN detector...")
            init_face_detector()
            # Run a dummy detection to allocate TF tensors
            face_detector.detect_faces(np.zeros((300, 300, 3), dtype=np.uint8))
            log.info("✅ MTCNN ready")
        except Exception as e:
            log.warning("MTCNN warmup failed (will retry on first request): %s", e)


def _embed(image_bgr):
    """Compute a face embedding for a BGR image array."""
    objs = DeepFace.represent(
        img_path=image_bgr,
        model_name=FACE_MODEL,
        enforce_detection=False,
        detector_backend="skip",
    )
    if objs and len(objs) > 0:
        return objs[0]["embedding"]
    return None


def load_student_photos():
    """Build embeddings for every student folder under STUDENT_PHOTOS_DIR."""
    global student_embeddings
    student_embeddings = {}

    if not os.path.isdir(STUDENT_PHOTOS_DIR):
        return False

    log.info("📸 Loading student photos with DeepFace (%s)...", FACE_MODEL)

    for student_folder in sorted(os.listdir(STUDENT_PHOTOS_DIR)):
        student_path = os.path.join(STUDENT_PHOTOS_DIR, student_folder)
        if not os.path.isdir(student_path):
            continue

        embeddings = []
        for photo_file in os.listdir(student_path):
            if not photo_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            photo_path = os.path.join(student_path, photo_file)
            try:
                img = cv2.imread(photo_path)
                if img is None:
                    log.warning("  Skipped %s/%s: unreadable", student_folder, photo_file)
                    continue
                # MTCNN detection + eye-based alignment
                faces = _detect_and_align(img)
                if not faces:
                    log.warning("  No face in %s/%s", student_folder, photo_file)
                    continue
                # Use the largest detected face (training photos should have one
                # primary subject)
                faces.sort(key=lambda t: t[1][2] * t[1][3], reverse=True)
                crop = faces[0][0]
                emb = _embed(crop)
                if emb is not None:
                    embeddings.append(emb)
            except Exception as e:
                log.warning("  Skipped %s/%s: %s", student_folder, photo_file, e)
                continue

        if embeddings:
            # Outlier check: if a student has 3+ photos and one is far from
            # all the others (> 0.70 mean cosine distance to peers), warn
            # the admin. This catches accidentally-uploaded photos of a
            # different person.
            if len(embeddings) >= 3:
                outliers = []
                for i, e in enumerate(embeddings):
                    others = [embeddings[j] for j in range(len(embeddings)) if j != i]
                    mean_d = float(np.mean([cosine_distance(e, o) for o in others]))
                    if mean_d > 0.70:
                        outliers.append((i, mean_d))
                if outliers:
                    log.warning(
                        "  ⚠️  %s: %d photo(s) look like a different person "
                        "(indices %s). Recognition still works (closest-match "
                        "matching), but consider replacing them.",
                        student_folder, len(outliers),
                        [i for i, _ in outliers],
                    )
            student_embeddings[student_folder] = embeddings
            log.info("  ✓ %s: %d photo(s) encoded", student_folder, len(embeddings))

    log.info("✅ Created embeddings for %d students", len(student_embeddings))
    return len(student_embeddings) > 0


def train_model():
    """Re-create embeddings from disk and persist them."""
    try:
        if not load_student_photos():
            log.warning("No student data found")
            return False
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(student_embeddings, f)
        log.info("💾 Saved embeddings for %d students", len(student_embeddings))
        return True
    except Exception as e:
        log.exception("Error training: %s", e)
        return False


def load_model():
    """Load embeddings from disk into memory."""
    global student_embeddings
    try:
        if os.path.exists(ENCODINGS_FILE):
            with open(ENCODINGS_FILE, "rb") as f:
                student_embeddings = pickle.load(f)
            log.info("✅ Loaded embeddings for %d students", len(student_embeddings))
            return True
        log.warning("Embeddings not found. Please train first.")
        return False
    except Exception as e:
        log.exception("Error loading embeddings: %s", e)
        return False


def cosine_distance(e1, e2):
    """Cosine distance between two embedding vectors."""
    a = np.array(e1)
    b = np.array(e2)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return float(1 - np.dot(a, b))


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------


def recognize_faces(rgb_image):
    """Detect faces in an RGB image and match them against known students.

    Detection is done by the standalone MTCNN package (with eye-based
    alignment), and the aligned crops are passed to Facenet for embedding.
    Embeddings are matched against the in-memory student_embeddings cache
    (loaded once at startup, no per-request disk I/O).
    """
    if not student_embeddings:
        load_model()
    if not student_embeddings:
        return {"recognized": [], "unknown": []}

    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    t_det = time.perf_counter()
    try:
        detections = _detect_and_align(bgr)
    except Exception as e:
        log.warning("Face detection failed: %s", e)
        detections = []
    detect_ms = (time.perf_counter() - t_det) * 1000
    log.info("🔍 Detected %d face(s) in %.0fms", len(detections), detect_ms)

    recognized, unknown = [], []

    t_emb = time.perf_counter()
    for crop, (x, y, w, h), conf in detections:
        embedding = _embed(crop)
        if embedding is None:
            continue
        location = {
            "top": int(y),
            "right": int(x + w),
            "bottom": int(y + h),
            "left": int(x),
        }

        # Score each student by their CLOSEST stored photo (min distance).
        # Min is much more robust than mean for face recognition: if any one
        # of the user's training photos strongly matches, that's enough
        # evidence — a single bad/mislabeled training photo can't drag the
        # whole student's score down. The margin check below still protects
        # against ambiguous matches between similar-looking strangers.
        per_student = []
        for name, emb_list in student_embeddings.items():
            dists = [cosine_distance(embedding, stored) for stored in emb_list]
            per_student.append((float(min(dists)), name))
        per_student.sort()  # ascending — closest first

        best_dist, best_name = per_student[0]
        second_dist = per_student[1][0] if len(per_student) > 1 else float("inf")
        margin = second_dist - best_dist
        confidence = 1.0 - best_dist

        # A face counts as "recognized" only when BOTH:
        #   1. The closest student is within RECOGNITION_THRESHOLD, AND
        #   2. The closest student is clearly better than the runner-up
        #      (margin check) — protects against ambiguous strangers.
        # When only one student is registered, margin is +inf (no runner-up).
        is_recognized = (
            best_dist < RECOGNITION_THRESHOLD
            and (margin >= RECOGNITION_MARGIN or len(per_student) == 1)
        )

        if is_recognized:
            log.info(
                "  ✅ %s (dist: %.3f, margin: %.3f, conf: %.2f%%)",
                best_name, best_dist, margin, confidence * 100,
            )
            existing = next(
                (s for s in recognized if s["student_id"] == best_name), None
            )
            if existing:
                if confidence > existing["confidence"]:
                    existing.update({
                        "confidence": float(confidence),
                        "raw_distance": float(best_dist),
                        "location": location,
                    })
            else:
                recognized.append({
                    "student_id": best_name,
                    "confidence": float(confidence),
                    "raw_distance": float(best_dist),
                    "location": location,
                })
        else:
            log.info(
                "  ❌ Unknown (closest: %s, dist: %.3f, margin: %.3f)",
                best_name, best_dist, margin,
            )
            unknown.append({
                "location": location,
                "distance": float(best_dist),
            })

    if detections:
        match_ms = (time.perf_counter() - t_emb) * 1000
        log.info(
            "🧠 Embedded+matched %d face(s) in %.0fms",
            len(detections), match_ms,
        )

    return {"recognized": recognized, "unknown": unknown}


def mark_attendance(student_ids):
    """Persist attendance for the given student ids in today's JSON file."""
    now = datetime.now(IST)
    today = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    attendance_file = os.path.join(ATTENDANCE_DIR, f"{today}.json")

    if os.path.exists(attendance_file):
        with open(attendance_file, "r") as f:
            data = json.load(f)
    else:
        data = {"date": today, "records": {}}

    for sid in student_ids:
        if sid not in data["records"]:
            data["records"][sid] = {"status": "Present", "timestamp": timestamp}

    with open(attendance_file, "w") as f:
        json.dump(data, f, indent=4)

    return data


# ---------------------------------------------------------------------------
# Routes — pages
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/attendance")
def attendance_page():
    return render_template("attendance.html")


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------


@app.route("/train", methods=["POST"])
def train():
    try:
        if train_model():
            return jsonify(
                {
                    "success": True,
                    "message": f"Created embeddings for {len(student_embeddings)} students using {FACE_MODEL}",
                    "count": len(student_embeddings),
                }
            )
        return jsonify({"success": False, "message": "No student data found"})
    except Exception as e:
        log.exception("Train failed")
        return jsonify({"success": False, "message": str(e)})


@app.route("/recognize", methods=["POST"])
def recognize():
    """
    Accepts JSON: {"image": "<dataURL or base64>", "mirror": false}
    Returns:     {"success": true, "recognized": [...], "unknown": [...], ...}
    """
    t_req = time.perf_counter()
    try:
        data = request.get_json(silent=True) or {}
        image_field = data.get("image")
        if not image_field:
            return jsonify({"success": False, "message": "No image provided"}), 400

        # Strip optional `data:image/...;base64,` prefix
        b64 = image_field.split(",", 1)[-1]
        image_bytes = base64.b64decode(b64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"success": False, "message": "Invalid image data"}), 400

        # Only flip when the client tells us this is a mirrored selfie capture
        if data.get("mirror", False):
            image = cv2.flip(image, 1)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = recognize_faces(rgb)
        recognized = result["recognized"]
        unknown = result["unknown"]

        response = {
            "success": True,
            # Mobile-friendly keys
            "recognized": recognized,
            "unknown": unknown,
            # Backwards-compatible keys for existing web client
            "recognized_students": recognized,
            "unknown_persons": unknown,
            "message": f"Found {len(recognized)} student(s)",
        }

        if recognized:
            ids = [s["student_id"] for s in recognized]
            response["attendance"] = mark_attendance(ids)

        if unknown:
            response["message"] += f" and {len(unknown)} unknown"
            response["alert"] = True

        total_ms = (time.perf_counter() - t_req) * 1000
        log.info(
            "⏱  /recognize took %.0fms (input %dx%d → recognized=%d, unknown=%d)",
            total_ms, image.shape[1], image.shape[0],
            len(recognized), len(unknown),
        )
        return jsonify(response)

    except Exception as e:
        log.exception("Recognize failed after %.0fms", (time.perf_counter() - t_req) * 1000)
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/attendance/today", methods=["GET"])
def get_today_attendance():
    try:
        today = datetime.now(IST).strftime("%Y-%m-%d")
        attendance_file = os.path.join(ATTENDANCE_DIR, f"{today}.json")
        if os.path.exists(attendance_file):
            with open(attendance_file, "r") as f:
                return jsonify({"success": True, "data": json.load(f)})
        return jsonify({"success": True, "data": {"date": today, "records": {}}})
    except Exception as e:
        log.exception("Today attendance failed")
        return jsonify({"success": False, "message": str(e)})


@app.route("/attendance/all", methods=["GET"])
def get_all_attendance():
    try:
        all_records = []
        for filename in os.listdir(ATTENDANCE_DIR):
            if filename.endswith(".json"):
                with open(os.path.join(ATTENDANCE_DIR, filename), "r") as f:
                    all_records.append(json.load(f))
        all_records.sort(key=lambda x: x["date"], reverse=True)
        return jsonify({"success": True, "data": all_records})
    except Exception as e:
        log.exception("All attendance failed")
        return jsonify({"success": False, "message": str(e)})


@app.route("/students", methods=["GET"])
def get_students():
    try:
        students = sorted(
            f
            for f in os.listdir(STUDENT_PHOTOS_DIR)
            if os.path.isdir(os.path.join(STUDENT_PHOTOS_DIR, f))
        )
        return jsonify({"success": True, "students": students, "count": len(students)})
    except Exception as e:
        log.exception("Students list failed")
        return jsonify({"success": False, "message": str(e)})


@app.route("/login", methods=["POST"])
def login():
    """Validate admin credentials sent by the mobile app."""
    try:
        data = request.get_json(silent=True) or {}
        if _check_admin(data):
            return jsonify({"success": True, "message": "Login successful"})
        return jsonify({"success": False, "message": "Invalid credentials"}), 401
    except Exception as e:
        log.exception("Login failed")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/add_student", methods=["POST"])
def add_student():
    """
    Register a new student from a mobile client.
    Accepts JSON: {"name": "student_name", "images": ["data:image/jpeg;base64,...", ...]}
    Saves images into student_photos/<name>/img_0.jpg, img_1.jpg, ...
    """
    try:
        data = request.get_json(silent=True) or {}

        # Admin authentication
        if not _check_admin(data):
            return jsonify({"success": False, "message": "Unauthorized"}), 401

        name = (data.get("name") or "").strip()
        images = data.get("images") or []

        if not name:
            return jsonify(
                {"success": False, "message": "Student name is required"}
            ), 400
        if not images or not isinstance(images, list):
            return jsonify(
                {"success": False, "message": "At least one image is required"}
            ), 400

        # Sanitize folder name to avoid path traversal
        safe_name = "".join(
            c for c in name if c.isalnum() or c in ("_", "-", " ")
        ).strip()
        if not safe_name:
            return jsonify({"success": False, "message": "Invalid student name"}), 400

        student_dir = os.path.join(STUDENT_PHOTOS_DIR, safe_name)
        os.makedirs(student_dir, exist_ok=True)

        # Continue numbering after any existing images
        existing = [
            f
            for f in os.listdir(student_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        start_idx = len(existing)

        saved = 0
        for i, image_str in enumerate(images):
            try:
                if not isinstance(image_str, str) or not image_str:
                    continue
                # Strip optional `data:image/...;base64,` header
                b64 = image_str.split(",", 1)[-1]
                image_bytes = base64.b64decode(b64)

                file_path = os.path.join(student_dir, f"img_{start_idx + saved}.jpg")
                with open(file_path, "wb") as f:
                    f.write(image_bytes)
                saved += 1
            except Exception as e:
                log.warning("Failed to save image %d for %s: %s", i, safe_name, e)
                continue

        if saved == 0:
            return jsonify(
                {"success": False, "message": "No valid images could be saved"}
            ), 400

        log.info("👤 Added %d photo(s) for student '%s'", saved, safe_name)
        return jsonify(
            {
                "success": True,
                "message": "Student added successfully",
                "name": safe_name,
                "saved": saved,
            }
        )

    except Exception as e:
        log.exception("add_student failed")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/export", methods=["GET"])
def export_attendance():
    """Export today's attendance as a downloadable Excel file."""
    try:
        date = request.args.get("date") or datetime.now(IST).strftime("%Y-%m-%d")
        attendance_file = os.path.join(ATTENDANCE_DIR, f"{date}.json")

        if not os.path.exists(attendance_file):
            return jsonify(
                {"success": False, "message": "No attendance data for this date"}
            ), 404

        with open(attendance_file, "r") as f:
            data = json.load(f)

        records = [
            {
                "Student": student,
                "Status": info.get("status", "Present"),
                "Time": info.get("timestamp", ""),
            }
            for student, info in data.get("records", {}).items()
        ]

        # Build Excel in-memory so we don't litter disk with stale exports
        df = pd.DataFrame(records, columns=["Student", "Status", "Time"])
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=date)
        buf.seek(0)

        return send_file(
            buf,
            as_attachment=True,
            download_name=f"attendance_{date}.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as e:
        log.exception("Export failed")
        return jsonify({"success": False, "message": str(e)}), 500


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

# Pre-load embeddings and warm the model at import time so the first request is fast.
load_model()
warmup_deepface()


if __name__ == "__main__":
    log.info("🚀 Starting Face Attendance System on http://0.0.0.0:5000")
    # use_reloader=False keeps a single TensorFlow process in memory
    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)
