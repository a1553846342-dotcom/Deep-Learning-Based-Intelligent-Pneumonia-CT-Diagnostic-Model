import tensorflow as tf
from PIL import Image, ImageTk
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from ttkthemes import ThemedTk 
import sys 
import glob # 引入 glob 用于查找文件夹中的图片

# --- 1. 配置参数 ---
MODEL_PATH = "model.tflite"
CLASSIFICATION_THRESHOLD = 0.65 

IMG_SIZE = 224
IMAGE_SHAPE = (IMG_SIZE, IMG_SIZE)
LABELS = ["NORMAL (正常)", "PNEUMONIA (肺炎)"]
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png'] # 定义支持的图片类型

COLOR_NORMAL = "#D4EDDA"        
COLOR_PNEUMONIA = "#F8D7DA"     
COLOR_DEFAULT_BG = "#e0e0e0"    

# --- 2. TFLite 推理核心函数 (保持不变) ---

interpreter_global = None
details_global = {}

def get_resource_path(relative_path):
    """获取打包后资源的正确绝对路径。"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)


def load_tflite_model(model_path):
    """加载 TFLite 解释器并获取反量化参数 (路径修正)"""
    global interpreter_global, details_global
    # ... (模型加载和参数提取代码保持不变) ...
    if interpreter_global is not None:
        return interpreter_global, details_global

    full_model_path = get_resource_path(model_path) 
    
    try:
        interpreter = tf.lite.Interpreter(model_path=full_model_path) 
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        quantization_params = output_details[0].get('quantization_parameters', {})
        scale = quantization_params.get('scales', [1.0])[0]
        zero_point = quantization_params.get('zero_points', [0])[0]
        
        details_global = {
            'input': input_details[0], 
            'output': output_details[0], 
            'scale': scale, 
            'zero_point': zero_point
        }
        interpreter_global = interpreter
        return interpreter, details_global
    except Exception as e:
        messagebox.showerror("模型错误", 
                             f"无法加载 TFLite 模型或获取参数: Could not open '{model_path}'.\n"
                             f"尝试加载路径: {full_model_path}\n"
                             f"错误信息: {e}")
        raise RuntimeError("Model Load Failed")

def preprocess_image(file_path):
    """针对量化模型，进行 uint8 预处理"""
    try:
        image = Image.open(file_path).convert('RGB').resize(IMAGE_SHAPE)
        image_array = np.asarray(image)
    except Exception:
        return None
    
    processed_data = image_array.astype(np.uint8)
    data = np.expand_dims(processed_data, axis=0)
    return data

def run_inference(file_path):
    """对单个图像运行 TFLite 推理并返回分类结果和概率"""
    
    try:
        interpreter, details = load_tflite_model(MODEL_PATH)
    except RuntimeError:
        return None, None

    input_data = preprocess_image(file_path)
    if input_data is None:
        return None, None

    try:
        interpreter.set_tensor(details['input']['index'], input_data)
        interpreter.invoke()
        quantized_output = interpreter.get_tensor(details['output']['index'])[0]
    except Exception:
        # 批量处理时，如果单个文件推理失败，只返回 None，不弹窗
        return None, None
    
    output = details['scale'] * (quantized_output.astype(np.float32) - details['zero_point'])

    pneumonia_prob = output[1] 
    
    if pneumonia_prob >= CLASSIFICATION_THRESHOLD:
        final_label = LABELS[1]
    else:
        final_label = LABELS[0]
        
    return final_label, pneumonia_prob

# --- 3. GUI 应用程序逻辑 (新增批量处理区域) ---

class PneumoniaDetectorApp:
    def __init__(self, master):
        self.master = master
        master.title(f"肺炎辅助诊断应用 (T={CLASSIFICATION_THRESHOLD:.2f})")

        self.current_file = None
        
        try:
            load_tflite_model(MODEL_PATH)
        except RuntimeError:
            pass 

        # ------------------- 主窗口布局 (可缩放) -------------------
        master.grid_columnconfigure(0, weight=1)
        master.grid_columnconfigure(1, weight=1)
        # 增加一行用于显示批量结果，并给它权重
        master.grid_rowconfigure(0, weight=1)
        master.grid_rowconfigure(1, weight=2) # 批量结果占据更多空间

        # ------------------- 左侧：文件选择/图片显示区域 (Row 0, Col 0) -------------------
        self.file_frame = ttk.Frame(master, padding="15") 
        self.file_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.file_frame.grid_columnconfigure(0, weight=1) 
        self.file_frame.grid_rowconfigure(0, weight=1) 

        # 图片预览 (Row 0)
        self.img_label = tk.Label(self.file_frame, text="[此处显示选中的胸片]", width=30, height=20, bg="white", borderwidth=2, relief="groove")
        self.img_label.grid(row=0, column=0, pady=15, sticky='nsew') 
        
        # 文件路径显示 (Row 1)
        self.path_label = ttk.Label(self.file_frame, text="未选择图片", foreground="gray")
        self.path_label.grid(row=1, column=0, pady=5)

        # 按钮容器 Frame (用于放置单个/批量选择按钮)
        self.button_container = ttk.Frame(self.file_frame)
        self.button_container.grid(row=2, column=0, pady=10)
        
        # 单文件选择按钮
        self.select_button = ttk.Button(self.button_container, text="📁 选择单个胸片", command=self.select_file)
        self.select_button.pack(side=tk.LEFT, padx=5)

        # ❗ 新增：批量文件夹选择按钮
        self.select_folder_button = ttk.Button(self.button_container, text="📂 批量选择文件夹", command=self.select_folder_and_run)
        self.select_folder_button.pack(side=tk.LEFT, padx=5)

        # ------------------- 右侧：单张推理/结果显示区域 (Row 0, Col 1) -------------------
        self.result_frame = ttk.Frame(master, padding="15")
        self.result_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        ttk.Label(self.result_frame, text="💡 单图诊断结果：", font=("Helvetica", 12, "bold")).pack(pady=10)

        self.result_label = tk.Label(self.result_frame, 
                                     text="等待分析...", 
                                     font=("Helvetica", 18, "bold"), 
                                     padx=20, pady=20, 
                                     relief="raised", 
                                     bg=COLOR_DEFAULT_BG)
        self.result_label.pack(pady=15, fill="both", expand=True) 

        self.prob_label = ttk.Label(self.result_frame, text="概率 (P_肺炎): N/A", font=("Helvetica", 10))
        self.prob_label.pack(pady=5)

        self.detect_button = ttk.Button(self.result_frame, 
                                         text="⚡️ 点击进行肺炎判断 (单张)", 
                                         command=self.run_detection, 
                                         state=tk.DISABLED, 
                                         style='Accent.TButton') 
        self.detect_button.pack(pady=10, fill="x", ipadx=10, ipady=5)


        # ------------------- 底部：批量结果显示区域 (Row 1, Col 0 & 1) -------------------
        self.batch_frame = ttk.Frame(master, padding="15")
        self.batch_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        self.batch_frame.grid_rowconfigure(1, weight=1)
        self.batch_frame.grid_columnconfigure(0, weight=1)
        
        ttk.Label(self.batch_frame, text="📋 批量检测结果：", font=("Helvetica", 12, "bold")).grid(row=0, column=0, sticky='w', pady=(0, 5))
        
        # Treeview 用于显示批量结果
        self.tree = self.create_batch_result_table(self.batch_frame)
        self.tree.grid(row=1, column=0, sticky='nsew')
        
        # 滚动条
        vsb = ttk.Scrollbar(self.batch_frame, orient="vertical", command=self.tree.yview)
        vsb.grid(row=1, column=1, sticky='ns')
        self.tree.configure(yscrollcommand=vsb.set)
        
    def create_batch_result_table(self, parent):
        """创建用于显示批量结果的 Treeview 表格"""
        columns = ("#1", "#2", "#3")
        tree = ttk.Treeview(parent, columns=columns, show="headings")
        
        tree.heading("#1", text="文件名")
        tree.heading("#2", text="诊断结果")
        tree.heading("#3", text="P(肺炎)")
        
        tree.column("#1", width=250, anchor='w')
        tree.column("#2", width=150, anchor='center')
        tree.column("#3", width=100, anchor='e')
        
        return tree

    # --- 文件选择和检测方法 ---

    def select_file(self):
        """选择单个文件"""
        self.tree.delete(*self.tree.get_children()) # 清空批量结果
        f_types = [('JPG/PNG Files', '*.jpg;*.png;*.jpeg')]
        file_path = filedialog.askopenfilename(filetypes=f_types)
        
        if file_path:
            self.current_file = file_path
            self.path_label.config(text=os.path.basename(file_path), foreground="black")
            
            try:
                img = Image.open(file_path)
                img = img.resize((200, 200), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(img)
                self.img_label.config(image=self.photo, text="")
                self.detect_button.config(state=tk.NORMAL)
                self.reset_results()
            except Exception as e:
                messagebox.showerror("图片错误", f"无法加载图片进行预览: {e}")
                self.reset_state()

    def select_folder_and_run(self):
        """选择文件夹并执行批量检测"""
        folder_path = filedialog.askdirectory()
        
        if folder_path:
            # 重置单图区域
            self.reset_state() 
            self.path_label.config(text=f"批量处理中: {os.path.basename(folder_path)}", foreground="blue")
            self.run_batch_detection(folder_path)

    def run_batch_detection(self, folder_path):
        """遍历文件夹中的所有图片并进行检测"""
        self.tree.delete(*self.tree.get_children()) # 清空旧结果
        image_files = []
        
        # 查找所有支持的图片文件
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            
        if not image_files:
            messagebox.showwarning("警告", f"在文件夹 '{os.path.basename(folder_path)}' 中未找到支持的图片文件。")
            self.path_label.config(text="批量检测完成", foreground="black")
            return

        total_files = len(image_files)
        success_count = 0
        
        for i, file_path in enumerate(image_files):
            file_name = os.path.basename(file_path)
            
            # 更新状态栏 (可选)
            self.path_label.config(text=f"批量处理中 ({i+1}/{total_files}): {file_name}", foreground="blue")
            self.master.update_idletasks() # 强制界面更新

            final_label, probability = run_inference(file_path)
            
            if final_label is not None:
                success_count += 1
                prob_str = f"{probability:.4f}"
                
                # 设置颜色标签
                if "PNEUMONIA" in final_label:
                    tag = 'pneumonia_tag'
                    display_label = "⚠️ 肺炎"
                else:
                    tag = 'normal_tag'
                    display_label = "✅ 正常"
                    
                # 插入结果到 Treeview
                self.tree.insert("", tk.END, values=(file_name, display_label, prob_str), tags=(tag,))
            else:
                self.tree.insert("", tk.END, values=(file_name, "处理失败", "N/A"), tags=('fail_tag',))

        # 配置 Treeview 标签颜色
        self.tree.tag_configure('normal_tag', background=COLOR_NORMAL)
        self.tree.tag_configure('pneumonia_tag', background=COLOR_PNEUMONIA)
        self.tree.tag_configure('fail_tag', background='#FFCCCC')
        
        messagebox.showinfo("批量检测完成", f"共处理 {total_files} 个文件，成功推理 {success_count} 个。")
        self.path_label.config(text=f"批量检测完成: {total_files} 个文件", foreground="black")


    def reset_results(self):
        """重置单图结果显示区域"""
        self.result_label.config(text="等待分析...", bg=COLOR_DEFAULT_BG)
        self.prob_label.config(text="概率 (P_肺炎): N/A")

    def reset_state(self):
        """重置所有状态和清空批量结果"""
        self.current_file = None
        self.path_label.config(text="未选择图片", foreground="gray")
        self.img_label.config(image=None, text="[此处显示选中的胸片]", bg="white")
        self.detect_button.config(state=tk.DISABLED)
        self.reset_results()
        self.tree.delete(*self.tree.get_children()) # 清空批量结果表格


    def run_detection(self):
        """执行单图推理"""
        if not self.current_file:
            messagebox.showwarning("警告", "请先选择一张图片。")
            return

        final_label, probability = run_inference(self.current_file)

        if final_label is None:
            self.reset_results()
            return

        self.prob_label.config(text=f"概率 (P_肺炎): {probability:.4f} (T={CLASSIFICATION_THRESHOLD:.2f})")
        
        if "PNEUMONIA" in final_label:
            bg_color = COLOR_PNEUMONIA
            display_text = "🚨 诊断: 肺炎 (PNEUMONIA)"
        else:
            bg_color = COLOR_NORMAL
            display_text = "🟢 诊断: 正常 (NORMAL)"

        self.result_label.config(text=display_text, bg=bg_color)


if __name__ == "__main__":
    root = ThemedTk(theme="arc") 
    root.geometry("800x650") # 扩大窗口以容纳批量结果
    
    app = PneumoniaDetectorApp(root)
    root.mainloop()