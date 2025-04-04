import tkinter as tk
from tkinter import filedialog, Label, Button, Text, Scrollbar, Canvas
from PIL import Image, ImageTk
import requests
import sqlite3
import os
import uuid
import base64
from io import BytesIO
import cv2
import numpy as np
import threading
import json

API_IMAGE_URL = "http://127.0.0.1:5000/detect/image/"
API_VIDEO_URL = "http://127.0.0.1:5000/detect/video/"
API_GET_IMAGES_URL = "http://127.0.0.1:5000/images/"  # API GET để lấy danh sách ảnh
API_DELETE_IMAGE_URL = "http://127.0.0.1:5000/images/"  # API DELETE để xóa ảnh
STATIC_DIR = "static"
BACKGROUND_IMAGE_PATH = "images/b.jpg"  # Thay đổi thành đường dẫn chính xác

class AppWithYolo:
    def __init__(self, root):
        self.root = root
        self.root.title("App lỏ by Tiến and Dương")
        self.root.geometry("1000x600")

        self.init_db()

        # Frame chính để chia bố cục
        self.main_frame = tk.Frame(root, bg="#f0f0f0")
        self.main_frame.pack(fill="both", expand=True)

        # Frame cho các nút (bên trái)
        self.button_frame = tk.Frame(self.main_frame, width=120, bg="#d9e6f2")
        self.button_frame.pack(side=tk.LEFT, fill="y", padx=(5, 0), pady=5)

        # Frame cho nội dung (canvas và kết quả)
        self.content_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.content_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=5, pady=5)

        # Tiêu đề
        self.label_title = Label(self.content_frame, text="App with Yolov8",
                                 font=("Arial", 16, "bold"), bg="#f0f0f0", fg="#333")
        self.label_title.pack(pady=(5, 2))

        # Frame cho canvas
        self.canvas_frame = tk.Frame(self.content_frame, bg="#f0f0f0")
        self.canvas_frame.pack(pady=2)

        self.canvas = Canvas(self.canvas_frame, width=800, height=400, bg="gray", highlightthickness=1,
                             highlightbackground="#ccc")
        self.canvas.pack(side=tk.LEFT)

        self.v_scrollbar = Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        self.v_scrollbar.pack(side=tk.RIGHT, fill="y")
        self.h_scrollbar = Scrollbar(self.content_frame, orient="horizontal", command=self.canvas.xview)
        self.h_scrollbar.pack(fill="x", pady=(0, 5))

        self.canvas.config(xscrollcommand=self.h_scrollbar.set, yscrollcommand=self.v_scrollbar.set)
        self.label_img = Label(self.canvas, bg="gray")
        self.canvas.create_window((0, 0), window=self.label_img, anchor="nw")

        # Lưu kích thước canvas để dùng cho background
        self.canvas_width = 800
        self.canvas_height = 400
        self.set_background_image()

        # Frame cho danh sách ảnh đã lưu (thumbnail)
        self.thumbnail_frame = tk.Frame(self.content_frame, bg="#f0f0f0")
        self.thumbnail_frame.pack(fill="x", pady=(2, 0))

        self.thumbnail_canvas = Canvas(self.thumbnail_frame, height=100, bg="#f0f0f0", highlightthickness=1,
                                       highlightbackground="#ccc")
        self.thumbnail_canvas.pack(fill="x", expand=True)

        self.thumbnail_scrollbar = Scrollbar(self.thumbnail_frame, orient="horizontal",
                                             command=self.thumbnail_canvas.xview)
        self.thumbnail_scrollbar.pack(fill="x")
        self.thumbnail_canvas.config(xscrollcommand=self.thumbnail_scrollbar.set)

        self.thumbnail_inner_frame = tk.Frame(self.thumbnail_canvas, bg="#f0f0f0")
        self.thumbnail_canvas.create_window((0, 0), window=self.thumbnail_inner_frame, anchor="nw")

        # Khu vực hiển thị kết quả nhận diện (bên dưới thumbnail)
        self.result_frame = tk.Frame(self.content_frame, bg="#f0f0f0")
        self.result_frame.pack(fill="x", pady=(5, 0))

        self.result_label = Label(self.result_frame, text="Ghi chú:", font=("Arial", 12, "bold"),
                                  bg="#f0f0f0", fg="#333")
        self.result_label.pack(anchor="w", pady=(0, 2))

        self.result_text = Text(self.result_frame, height=4, width=100, font=("Arial", 10),
                                wrap="word", borderwidth=1, relief="solid")
        self.result_text.pack(side=tk.LEFT, fill="x", expand=True)

        self.scrollbar = Scrollbar(self.result_frame, command=self.result_text.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=self.scrollbar.set)

        # Frame cho các nút điều hướng (Ảnh trước, Ảnh sau, Xóa)
        self.nav_frame = tk.Frame(self.content_frame, bg="#f0f0f0")
        self.nav_frame.pack(fill="x", pady=(2, 0))

        self.btn_prev = Button(self.nav_frame, text="Ảnh trước", font=("Arial", 10),
                               bg="#FF5722", fg="white", activebackground="#e64a19",
                               command=self.show_prev_image, state="disabled")
        self.btn_prev.pack(side=tk.LEFT, padx=5)

        self.btn_next = Button(self.nav_frame, text="Ảnh sau", font=("Arial", 10),
                               bg="#FF5722", fg="white", activebackground="#e64a19",
                               command=self.show_next_image, state="disabled")
        self.btn_next.pack(side=tk.LEFT, padx=5)

        # Nút Xóa
        self.btn_delete = Button(self.nav_frame, text="Xóa ảnh", font=("Arial", 10),
                                 bg="#F44336", fg="white", activebackground="#d32f2f",
                                 command=self.delete_image, state="disabled")
        self.btn_delete.pack(side=tk.LEFT, padx=5)

        # Nút Dừng Video (ban đầu ẩn)
        self.btn_stop_video = Button(self.nav_frame, text="Dừng Video", font=("Arial", 10),
                                     bg="#F44336", fg="white", activebackground="#d32f2f",
                                     command=self.stop_video)
        self.btn_stop_video.pack(side=tk.LEFT, padx=5)
        self.btn_stop_video.pack_forget()  # Ẩn nút này ban đầu

        # Các nút chức năng xếp dọc trong button_frame
        self.btn_select_image = Button(self.button_frame, text="Chọn Ảnh", font=("Arial", 11),
                                       bg="#4CAF50", fg="white", activebackground="#45a049",
                                       command=self.select_image)
        self.btn_select_image.pack(fill="x", pady=2, padx=5)

        self.btn_select_video = Button(self.button_frame, text="Chọn Video", font=("Arial", 11),
                                       bg="#2196F3", fg="white", activebackground="#1e88e5",
                                       command=self.select_video)
        self.btn_select_video.pack(fill="x", pady=2, padx=5)

        self.btn_capture = Button(self.button_frame, text="Lưu ảnh", font=("Arial", 11),
                                  bg="#FF9800", fg="white", activebackground="#fb8c00",
                                  command=self.capture_image)
        self.btn_capture.pack(fill="x", pady=2, padx=5)
        self.btn_capture.config(state="disabled")

        self.btn_view_captured = Button(self.button_frame, text="Xem ảnh đã lưu", font=("Arial", 11),
                                        bg="#9C27B0", fg="white", activebackground="#8e24aa",
                                        command=self.view_captured_images)
        self.btn_view_captured.pack(fill="x", pady=2, padx=5)

        # Nút Live Camera
        self.btn_live_camera = Button(self.button_frame, text="Live Camera", font=("Arial", 11),
                                      bg="#F44336", fg="white", activebackground="#d32f2f",
                                      command=self.start_live_camera)
        self.btn_live_camera.pack(fill="x", pady=2, padx=5)

        # Nút Dừng Camera (ban đầu ẩn)
        self.btn_stop_camera = Button(self.button_frame, text="Dừng Camera", font=("Arial", 11),
                                      bg="#F44336", fg="white", activebackground="#d32f2f",
                                      command=self.stop_live_camera)
        self.btn_stop_camera.pack(fill="x", pady=2, padx=5)
        self.btn_stop_camera.pack_forget()  # Ẩn nút này ban đầu

        # Biến lưu dữ liệu ảnh hiện tại (base64)
        self.current_image_data = None
        # Biến lưu danh sách ảnh đã lưu và chỉ số ảnh hiện tại
        self.captured_images = []
        self.current_image_index = -1
        # Danh sách để lưu tham chiếu đến các thumbnail
        self.thumbnail_labels = []
        # Biến để kiểm soát live camera và video
        self.camera_running = False
        self.video_running = False
        self.cap = None
        # Biến lưu detections hiện tại để sử dụng khi chụp lại
        self.current_detections = []

    def init_db(self):
        conn = sqlite3.connect("captured_images.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                detections TEXT
            )
        """)
        conn.commit()
        conn.close()

    def set_background_image(self):
        try:
            img = Image.open(BACKGROUND_IMAGE_PATH)
            original_width, original_height = img.size

            canvas_ratio = self.canvas_width / self.canvas_height
            image_ratio = original_width / original_height

            if image_ratio > canvas_ratio:
                new_height = self.canvas_height
                new_width = int(new_height * image_ratio)
            else:
                new_width = self.canvas_width
                new_height = int(new_width / image_ratio)

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            self.background_img_tk = ImageTk.PhotoImage(img)
            self.label_img.config(image=self.background_img_tk)
            self.canvas.config(scrollregion=(0, 0, new_width, new_height))
        except Exception as e:
            self.label_img.config(image=None, text="Chưa có nội dung")

    def start_live_camera(self):
        if self.camera_running or self.video_running:
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Không thể mở camera!")
            return

        self.camera_running = True
        self.btn_live_camera.pack_forget()
        self.btn_stop_camera.pack(fill="x", pady=2, padx=5)
        self.btn_select_image.config(state="disabled")
        self.btn_select_video.config(state="disabled")
        self.btn_view_captured.config(state="disabled")
        self.btn_prev.config(state="disabled")
        self.btn_next.config(state="disabled")
        self.btn_delete.config(state="disabled")
        self.clear_thumbnails()

        self.thread = threading.Thread(target=self.process_camera)
        self.thread.daemon = True
        self.thread.start()

    def stop_live_camera(self):
        if not self.camera_running:
            return

        self.camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_stop_camera.pack_forget()
        self.btn_live_camera.pack(fill="x", pady=2, padx=5)
        self.btn_select_image.config(state="normal")
        self.btn_select_video.config(state="normal")
        self.btn_capture.config(state="normal")
        self.btn_view_captured.config(state="normal")
        self.set_background_image()
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Camera đã dừng.")

    def process_camera(self):
        while self.camera_running:
            ret, frame = self.cap.read()
            if not ret:
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "Không thể đọc frame từ camera!")
                break

            _, buffer = cv2.imencode(".jpg", frame)
            files = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
            try:
                response = requests.post(API_IMAGE_URL, files=files)
                if response.status_code == 200:
                    data = response.json()
                    image_base64 = data["image_data"]
                    detections = data.get("detections", [])
                    self.current_image_data = image_base64
                    self.current_detections = detections
                    self.show_image_from_base64(image_base64)
                    self.show_detections(detections, "camera")
                    self.btn_capture.config(state="normal")
                else:
                    self.result_text.delete(1.0, tk.END)
                    self.result_text.insert(tk.END, f"Lỗi từ API: {response.status_code} - {response.text}")
                    self.btn_capture.config(state="disabled")
            except Exception as e:
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"Lỗi: {str(e)}")
                self.btn_capture.config(state="disabled")

            self.root.update()
            cv2.waitKey(30)

        if self.cap:
            self.cap.release()

    def select_image(self):
        if self.camera_running or self.video_running:
            return
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if not file_path:
            return
        self.process_file(file_path, API_IMAGE_URL, "ảnh")

    def select_video(self):
        if self.camera_running or self.video_running:
            return
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if not file_path:
            return
        self.play_video(file_path)

    def play_video(self, file_path):
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Không thể mở file video!")
            return

        self.video_running = True
        self.btn_stop_video.pack(side=tk.LEFT, padx=5)
        self.btn_select_image.config(state="disabled")
        self.btn_select_video.config(state="disabled")
        self.btn_view_captured.config(state="disabled")
        self.btn_prev.config(state="disabled")
        self.btn_next.config(state="disabled")
        self.btn_delete.config(state="disabled")
        self.btn_live_camera.config(state="disabled")
        self.clear_thumbnails()

        self.thread = threading.Thread(target=self.process_video)
        self.thread.daemon = True
        self.thread.start()

    def stop_video(self):
        if not self.video_running:
            return

        self.video_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_stop_video.pack_forget()
        self.btn_select_image.config(state="normal")
        self.btn_select_video.config(state="normal")
        self.btn_capture.config(state="normal")
        self.btn_view_captured.config(state="normal")
        self.btn_live_camera.config(state="normal")
        self.set_background_image()
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Video đã dừng.")

    def process_video(self):
        while self.video_running:
            ret, frame = self.cap.read()
            if not ret:
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "Đã phát hết video hoặc lỗi khi đọc frame!")
                self.stop_video()
                break

            _, buffer = cv2.imencode(".jpg", frame)
            files = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
            try:
                response = requests.post(API_IMAGE_URL, files=files)
                if response.status_code == 200:
                    data = response.json()
                    image_base64 = data["image_data"]
                    detections = data.get("detections", [])
                    self.current_image_data = image_base64
                    self.current_detections = detections
                    self.show_image_from_base64(image_base64)
                    self.show_detections(detections, "video")
                    self.btn_capture.config(state="normal")
                else:
                    self.result_text.delete(1.0, tk.END)
                    self.result_text.insert(tk.END, f"Lỗi từ API: {response.status_code} - {response.text}")
                    self.btn_capture.config(state="disabled")
            except Exception as e:
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"Lỗi: {str(e)}")
                self.btn_capture.config(state="disabled")

            self.root.update()
            cv2.waitKey(30)

        if self.cap:
            self.cap.release()

    def process_file(self, file_path, api_url, file_type):
        try:
            with open(file_path, "rb") as file:
                response = requests.post(api_url, files={"file": file})

            if response.status_code == 200:
                data = response.json()
                self.current_image_data = data["image_data"]
                self.current_detections = data.get("detections", [])
                self.show_image_from_base64(self.current_image_data)
                self.show_detections(self.current_detections, file_type)
                self.btn_capture.config(state="normal")
                self.btn_prev.config(state="disabled")
                self.btn_next.config(state="disabled")
                self.btn_delete.config(state="disabled")
                self.clear_thumbnails()
            else:
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"Lỗi từ API: {response.status_code} - {response.text}")
                self.btn_capture.config(state="disabled")
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Lỗi: {str(e)}")
            self.btn_capture.config(state="disabled")

    def show_image_from_base64(self, image_base64):
        try:
            image_data = base64.b64decode(image_base64)
            img = Image.open(BytesIO(image_data))
            original_width, original_height = img.size

            canvas_ratio = self.canvas_width / self.canvas_height
            image_ratio = original_width / original_height

            if image_ratio > canvas_ratio:
                new_height = self.canvas_height
                new_width = int(new_height * image_ratio)
            else:
                new_width = self.canvas_width
                new_height = int(new_width / image_ratio)

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            img_tk = ImageTk.PhotoImage(img)
            self.label_img.config(image=img_tk, text="")
            self.label_img.image = img_tk
            self.canvas.config(scrollregion=(0, 0, new_width, new_height))
        except Exception as e:
            self.label_img.config(image=None, text=f"Lỗi tải nội dung: {str(e)}")

    def show_detections(self, detections, file_type):
        self.result_text.delete(1.0, tk.END)
        if not detections:
            self.result_text.insert(tk.END, f"Không phát hiện được đối tượng nào trong {file_type}.")
            return

        header = f"Kết quả nhận diện từ {file_type}:\n\n"
        self.result_text.insert(tk.END, header)

        for idx, detection in enumerate(detections, 1):
            label = detection["label"]
            confidence = detection["confidence"]
            box = detection["box"]
            result = (
                f"Đối tượng {idx}:\n"
                f"  - Nhãn: {label}\n"
                f"  - Độ tin cậy: {confidence:.2f}\n"
                f"  - Vị trí: (x1: {box[0]}, y1: {box[1]}, x2: {box[2]}, y2: {box[3]})\n\n"
            )
            self.result_text.insert(tk.END, result)

    def capture_image(self):
        if not self.current_image_data:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Không có ảnh để chụp lại!")
            return

        new_filename = f"captured_{uuid.uuid4().hex}.jpg"
        new_path = os.path.join(STATIC_DIR, new_filename)

        try:
            image_data = base64.b64decode(self.current_image_data)
            img = Image.open(BytesIO(image_data))
            img.save(new_path)

            detections_json = json.dumps(self.current_detections)

            conn = sqlite3.connect("captured_images.db")
            cursor = conn.cursor()
            cursor.execute("INSERT INTO images (file_path, detections) VALUES (?, ?)",
                           (new_path, detections_json))
            conn.commit()
            image_id = cursor.lastrowid
            conn.close()

            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Đã lưu ảnh với ID: {image_id}\nĐường dẫn: {new_path}")
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Lỗi khi lưu ảnh: {str(e)}")

    def view_captured_images(self):
        if self.camera_running or self.video_running:
            return
        try:
            # Gọi API GET để lấy danh sách ảnh
            response = requests.get(API_GET_IMAGES_URL)
            if response.status_code == 200:
                data = response.json()
                images = data.get("images", [])

                self.result_text.delete(1.0, tk.END)
                if not images:
                    self.result_text.insert(tk.END, "Chưa có ảnh nào được lưu.")
                    self.set_background_image()
                    self.btn_prev.config(state="disabled")
                    self.btn_next.config(state="disabled")
                    self.btn_delete.config(state="disabled")
                    self.clear_thumbnails()
                    return

                # Lưu danh sách ảnh đã lưu
                self.captured_images = [(img["id"], img["file_path"], img["detections"]) for img in images]
                self.current_image_index = 0

                # Hiển thị danh sách thumbnail
                self.display_thumbnails()

                # Hiển thị ảnh đầu tiên
                self.show_captured_image()
                # Kích hoạt nút điều hướng
                self.update_navigation_buttons()
            else:
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"Lỗi từ API: {response.status_code} - {response.text}")
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Lỗi khi gọi API: {str(e)}")

    def delete_image(self):
        if self.current_image_index < 0 or self.current_image_index >= len(self.captured_images):
            return

        image_id, _, _ = self.captured_images[self.current_image_index]
        try:
            # Gọi API DELETE để xóa ảnh
            response = requests.delete(f"{API_DELETE_IMAGE_URL}{image_id}")
            if response.status_code == 200:
                # Xóa ảnh khỏi danh sách và cập nhật giao diện
                self.captured_images.pop(self.current_image_index)
                if not self.captured_images:
                    self.result_text.delete(1.0, tk.END)
                    self.result_text.insert(tk.END, "Chưa có ảnh nào được lưu.")
                    self.set_background_image()
                    self.btn_prev.config(state="disabled")
                    self.btn_next.config(state="disabled")
                    self.btn_delete.config(state="disabled")
                    self.clear_thumbnails()
                    self.current_image_index = -1
                    return

                # Điều chỉnh chỉ số ảnh hiện tại
                if self.current_image_index >= len(self.captured_images):
                    self.current_image_index = len(self.captured_images) - 1

                # Cập nhật giao diện
                self.display_thumbnails()
                self.show_captured_image()
                self.update_navigation_buttons()
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"Đã xóa ảnh với ID: {image_id}")
            else:
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"Lỗi từ API: {response.status_code} - {response.text}")
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Lỗi khi gọi API: {str(e)}")

    def display_thumbnails(self):
        self.clear_thumbnails()

        x_position = 5
        thumbnail_size = (80, 60)
        for idx, (image_id, file_path, _) in enumerate(self.captured_images):
            try:
                img = Image.open(file_path)
                img = img.resize(thumbnail_size, Image.Resampling.LANCZOS)
                img_tk = ImageTk.PhotoImage(img)

                thumb_frame = tk.Frame(self.thumbnail_inner_frame, bg="#f0f0f0")
                thumb_frame.grid(row=0, column=idx, padx=5)

                thumb_label = Label(thumb_frame, image=img_tk, bg="#f0f0f0")
                thumb_label.image = img_tk
                thumb_label.pack()

                id_label = Label(thumb_frame, text=f"ID: {image_id}", font=("Arial", 8), bg="#f0f0f0", fg="#333")
                id_label.pack()

                thumb_label.bind("<Button-1>", lambda event, index=idx: self.select_thumbnail(index))

                self.thumbnail_labels.append((thumb_frame, thumb_label, id_label))
            except Exception as e:
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"Lỗi khi tải ảnh ID {image_id}: {str(e)}")

        self.thumbnail_inner_frame.update_idletasks()
        self.thumbnail_canvas.config(scrollregion=self.thumbnail_canvas.bbox("all"))

    def clear_thumbnails(self):
        for thumb_frame, _, _ in self.thumbnail_labels:
            thumb_frame.destroy()
        self.thumbnail_labels = []
        self.thumbnail_canvas.config(scrollregion=(0, 0, 0, 0))

    def select_thumbnail(self, index):
        self.current_image_index = index
        self.show_captured_image()
        self.update_navigation_buttons()

    def show_captured_image(self):
        if self.current_image_index < 0 or self.current_image_index >= len(self.captured_images):
            return

        image_id, file_path, detections = self.captured_images[self.current_image_index]
        try:
            img = Image.open(file_path)
            original_width, original_height = img.size

            canvas_ratio = self.canvas_width / self.canvas_height
            image_ratio = original_width / original_height

            if image_ratio > canvas_ratio:
                new_height = self.canvas_height
                new_width = int(new_height * image_ratio)
            else:
                new_width = self.canvas_width
                new_height = int(new_width / image_ratio)

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            img_tk = ImageTk.PhotoImage(img)
            self.label_img.config(image=img_tk, text="")
            self.label_img.image = img_tk
            self.canvas.config(scrollregion=(0, 0, new_width, new_height))

            self.show_detections(detections, "ảnh đã lưu")
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Lỗi khi tải ảnh: {str(e)}")

    def show_prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_captured_image()
            self.update_navigation_buttons()

    def show_next_image(self):
        if self.current_image_index < len(self.captured_images) - 1:
            self.current_image_index += 1
            self.show_captured_image()
            self.update_navigation_buttons()

    def update_navigation_buttons(self):
        self.btn_prev.config(state="normal" if self.current_image_index > 0 else "disabled")
        self.btn_next.config(state="normal" if self.current_image_index < len(self.captured_images) - 1 else "disabled")
        self.btn_delete.config(state="normal" if self.captured_images else "disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = AppWithYolo(root)
    root.mainloop()