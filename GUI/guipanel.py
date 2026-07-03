import os
import shutil
import sys
import threading
from datetime import datetime
from typing import Dict, List, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk, ImageOps, ImageChops
import torch

# Project root for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
	sys.path.append(PROJECT_ROOT)

from inference.denoise import load_config, load_image, infer_image, save_output
try:
	from evaluate.metrics import psnr, ssim_batch
	METRICS_AVAILABLE = True
except Exception:
	psnr = None
	ssim_batch = None
	METRICS_AVAILABLE = False
from models.hybrid_model import HybridDenoiser
from scripts.moderate_denoise import moderate_denoise


class DenoiseApp:
	def __init__(self, root: tk.Tk) -> None:
		self.root = root
		self.root.title("Hybrid CNN-Transformer Denoising")
		self.root.geometry("1100x720")
		self.root.configure(bg="#eef1f7")

		self.preview_w = 420
		self.preview_h = 280

		self.input_image_path: Optional[str] = None
		self.output_image_path: Optional[str] = None
		self.base_output_image_path: Optional[str] = None
		self.comparison_image_path: Optional[str] = None
		self.heatmap_image_path: Optional[str] = None

		self.input_photo = None
		self.output_photo = None
		self._image_refs: Dict[tk.Label, ImageTk.PhotoImage] = {}
		self.prompt_text = None
		self.prompt_status_label = None
		self.prompt_settings = None
		self.use_prompt_var = tk.BooleanVar(value=True)

		self.model = None
		self.device = None
		self.cfg = None
		self.last_metrics = {}

		self._init_style()
		self._build_layout()
		self._load_model()

	def _init_style(self) -> None:
		style = ttk.Style(self.root)
		style.theme_use("clam")

		self.colors = {
			"bg": "#eef1f7",
			"card": "#f6f8ff",
			"shadow": "#d6dceb",
			"accent": "#9cc5ff",
			"accent_dark": "#7aa9f0",
			"text": "#1b1f2a",
			"muted": "#5f6b85",
			"panel": "#edf1fb",
		}

		self.root.configure(bg=self.colors["bg"])
		style.configure("TFrame", background=self.colors["bg"])
		style.configure("Card.TLabelframe", background=self.colors["card"], foreground=self.colors["text"])
		style.configure("Card.TLabelframe.Label", background=self.colors["card"], foreground=self.colors["text"])
		style.configure("Header.TLabel", background=self.colors["bg"], foreground=self.colors["text"], font=("Segoe UI", 16, "bold"))
		style.configure("Sub.TLabel", background=self.colors["bg"], foreground=self.colors["muted"], font=("Segoe UI", 10))
		style.configure("TButton", background=self.colors["accent"], foreground="#0f1b33", padding=(12, 8), relief="flat")
		style.map("TButton", background=[("active", self.colors["accent_dark"])])

	def _build_layout(self) -> None:
		container = ttk.Frame(self.root, padding=16)
		container.pack(fill=tk.BOTH, expand=True)

		header = ttk.Label(container, text="Image Denoising Panel", style="Header.TLabel")
		header.pack(anchor=tk.W)

		subtitle = ttk.Label(
			container,
			text="Upload a noisy image, run denoise, and review results.",
			style="Sub.TLabel",
		)
		subtitle.pack(anchor=tk.W, pady=(2, 12))

		top_row = ttk.Frame(container)
		top_row.pack(fill=tk.X)

		input_wrap, self.input_frame = self._make_card(top_row, "Input (Noisy)")
		input_wrap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
		input_wrap.configure(width=360, height=320)
		input_wrap.pack_propagate(False)

		heatmap_wrap, self.heatmap_frame = self._make_card(top_row, "Heatmap")
		heatmap_wrap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 8))
		heatmap_wrap.configure(width=360, height=320)
		heatmap_wrap.pack_propagate(False)

		output_wrap, self.output_frame = self._make_card(top_row, "Output (Denoised)")
		output_wrap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))
		output_wrap.configure(width=360, height=320)
		output_wrap.pack_propagate(False)

		self.input_image_label = tk.Label(self.input_frame, bg=self.colors["panel"], width=60, height=18)
		self.input_image_label.pack(fill=tk.BOTH, expand=True)

		self.heatmap_image_label = tk.Label(self.heatmap_frame, bg=self.colors["panel"], width=60, height=18)
		self.heatmap_image_label.pack(fill=tk.BOTH, expand=True)

		self.output_image_label = tk.Label(self.output_frame, bg=self.colors["panel"], width=60, height=18)
		self.output_image_label.pack(fill=tk.BOTH, expand=True)

		button_row = ttk.Frame(container)
		button_row.pack(fill=tk.X, pady=12)

		self.upload_btn = ttk.Button(button_row, text="Upload Noisy Image", command=self.on_upload)
		self.upload_btn.pack(side=tk.LEFT)

		self.denoise_btn = ttk.Button(button_row, text="Start Denoise", command=self.on_denoise)
		self.denoise_btn.pack(side=tk.LEFT, padx=10)

		self.save_btn = ttk.Button(button_row, text="Save Result As", command=self.on_save_as)
		self.save_btn.pack(side=tk.LEFT)
		self.save_btn.configure(state=tk.DISABLED)

		prompt_row = ttk.Frame(container)
		prompt_row.pack(fill=tk.X, pady=(0, 12))

		prompt_wrap, prompt_frame = self._make_card(prompt_row, "Prompt Tuning")
		prompt_wrap.pack(fill=tk.BOTH, expand=True)
		prompt_wrap.configure(height=120)
		prompt_wrap.pack_propagate(False)

		prompt_toolbar = ttk.Frame(prompt_frame)
		prompt_toolbar.pack(fill=tk.X, pady=(0, 6))

		prompt_toggle = ttk.Checkbutton(
			prompt_toolbar,
			text="Use prompt-based tuning",
			variable=self.use_prompt_var,
		)
		prompt_toggle.pack(side=tk.LEFT)

		self.prompt_status_label = ttk.Label(prompt_toolbar, text="", style="Sub.TLabel")
		self.prompt_status_label.pack(side=tk.RIGHT)

		self.prompt_text = tk.Text(prompt_frame, height=4, wrap=tk.WORD, bg=self.colors["card"], fg=self.colors["text"], relief="flat")
		self.prompt_text.pack(fill=tk.BOTH, expand=True)

		default_prompt = (
			"Remove extreme multicolor digital noise and heavy grain from this image while preserving all original details and textures. "
			"Maintain the natural colors, lighting, and depth of the scene. Apply intelligent noise reduction (40–60%) without oversmoothing. "
			"Keep edges sharp and well-defined. Preserve fine structural details and surface textures. Improve clarity and contrast slightly while keeping the image realistic and photo-accurate. "
			"Avoid artificial smoothing, plastic look, or fake detail generation. Ensure the final result looks natural and professionally restored."
		)
		self.prompt_text.insert("1.0", default_prompt)

		bottom_row = ttk.Frame(container)
		bottom_row.pack(fill=tk.BOTH, expand=True)

		log_wrap, self.log_frame = self._make_card(bottom_row, "Processing Log")
		log_wrap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
		log_wrap.configure(width=520, height=240)
		log_wrap.pack_propagate(False)

		metrics_wrap, self.metrics_frame = self._make_card(bottom_row, "Result Summary")
		metrics_wrap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))
		metrics_wrap.configure(width=520, height=240)
		metrics_wrap.pack_propagate(False)

		log_inner = ttk.Frame(self.log_frame)
		log_inner.pack(fill=tk.BOTH, expand=True)

		self.log_text = tk.Text(log_inner, height=10, wrap=tk.WORD, bg=self.colors["card"], fg=self.colors["text"], relief="flat")
		self.log_text.configure(state=tk.DISABLED)
		self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

		log_scroll = ttk.Scrollbar(log_inner, orient=tk.VERTICAL, command=self.log_text.yview)
		log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
		self.log_text.configure(yscrollcommand=log_scroll.set)

		metrics_inner = ttk.Frame(self.metrics_frame)
		metrics_inner.pack(fill=tk.BOTH, expand=True)

		self.metrics_text = tk.Text(metrics_inner, height=10, wrap=tk.WORD, bg=self.colors["card"], fg=self.colors["text"], relief="flat")
		self.metrics_text.configure(state=tk.DISABLED)
		self.metrics_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

		metrics_scroll = ttk.Scrollbar(metrics_inner, orient=tk.VERTICAL, command=self.metrics_text.yview)
		metrics_scroll.pack(side=tk.RIGHT, fill=tk.Y)
		self.metrics_text.configure(yscrollcommand=metrics_scroll.set)

		self._set_metrics_text("No results yet.")

	def _make_card(self, parent: ttk.Frame, title: str):
		shadow = tk.Frame(parent, bg=self.colors["shadow"], bd=0)
		card_container = tk.Frame(shadow, bg=self.colors["shadow"], bd=0)
		card_container.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

		card = ttk.Labelframe(card_container, text=title, style="Card.TLabelframe", padding=12)
		card.pack(fill=tk.BOTH, expand=True)
		return shadow, card

	def _load_model(self) -> None:
		try:
			cfg_path = os.path.join(PROJECT_ROOT, "config.yaml")
			self.cfg = load_config(cfg_path)

			use_gpu = self.cfg.get("device", {}).get("use_gpu", True)
			gpu_id = self.cfg.get("device", {}).get("gpu_id", 0)

			if use_gpu and torch.cuda.is_available():
				self.device = torch.device(f"cuda:{gpu_id}")
			else:
				self.device = torch.device("cpu")

			model_cfg = self.cfg.get("model", {})
			self.model = HybridDenoiser(
				in_channels=3,
				base_channels=model_cfg.get("base_channels", 48),
				num_cnn_blocks=model_cfg.get("num_cnn_blocks", 17),
				num_transformer_blocks=model_cfg.get("num_transformer_blocks", [2, 2, 4, 4]),
				num_heads=model_cfg.get("num_heads", [1, 2, 4, 8]),
				expansion_factor=model_cfg.get("expansion_factor", 2.0),
				fusion_type=model_cfg.get("fusion_type", "concat"),
				residual_learning=model_cfg.get("use_residual_learning", True),
			).to(self.device)

			ckpt_path = os.path.join(PROJECT_ROOT, "experiments", "checkpoints", "hybrid_best.pth")
			ckpt = torch.load(ckpt_path, map_location=self.device)
			self.model.load_state_dict(ckpt["model_state_dict"])

			self._log(f"Model loaded on {self.device}.")
		except Exception as exc:
			self._log(f"Failed to load model: {exc}")
			self.denoise_btn.configure(state=tk.DISABLED)
			self.save_btn.configure(state=tk.DISABLED)

	def on_upload(self) -> None:
		filetypes = [
			("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
			("All files", "*.*"),
		]
		path = filedialog.askopenfilename(title="Select noisy image", filetypes=filetypes)
		if not path:
			return

		self.input_image_path = path
		self._show_image(path, self.input_image_label, is_input=True)
		self._log(f"Loaded input image: {os.path.basename(path)}")

	def on_denoise(self) -> None:
		if not self.input_image_path:
			messagebox.showwarning("Missing image", "Please upload a noisy image first.")
			return

		if self.model is None:
			messagebox.showerror("Model not loaded", "Model failed to load. Check logs.")
			return

		self.denoise_btn.configure(state=tk.DISABLED)
		self.save_btn.configure(state=tk.DISABLED)
		self.prompt_settings = self._get_prompt_settings()
		self._log("Starting denoise...")

		worker = threading.Thread(target=self._run_denoise, daemon=True)
		worker.start()

	def _run_denoise(self) -> None:
		try:
			image_tensor = load_image(self.input_image_path, resize=None)
			image_tensor_device = image_tensor.to(self.device)

			output = self._infer_tiled(image_tensor_device)
			output_cpu = output.detach().cpu()

			results_dir = os.path.join(PROJECT_ROOT, "results")
			os.makedirs(results_dir, exist_ok=True)
			timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
			self.output_image_path = os.path.join(results_dir, f"denoised_{timestamp}.png")
			self.base_output_image_path = self.output_image_path
			self.comparison_image_path = os.path.join(results_dir, f"comparison_{timestamp}.png")
			self.heatmap_image_path = os.path.join(results_dir, f"heatmap_{timestamp}.png")

			save_output(output_cpu, self.output_image_path)

			improve_path = os.path.join(results_dir, f"denoised_improved_{timestamp}.png")
			improved = moderate_denoise(
				self.output_image_path,
				improve_path,
				strength=self._get_prompt_value("post_strength", 0.45),
				clarity=self._get_prompt_value("clarity", 0.12),
				contrast=self._get_prompt_value("contrast", 0.1),
				sharpen=self._get_prompt_value("sharpen", 0.18),
			)
			if improved:
				self.output_image_path = improve_path
				self._log(f"Improved output saved: {os.path.basename(improve_path)}")
			else:
				self._log("Postprocess skipped (OpenCV/PIL not available or failed).")

			self._save_comparison_image(self.comparison_image_path)

			noise_est = (image_tensor - output_cpu).abs().mean().item()
			noise_pct = max(0.0, min(100.0, noise_est * 100.0))
			strength = noise_est / (image_tensor.abs().mean().item() + 1e-8)
			strength_pct = max(0.0, min(100.0, strength * 100.0))

			psnr_val = None
			ssim_val = None
			if METRICS_AVAILABLE and psnr is not None and ssim_batch is not None:
				try:
					psnr_val = psnr(output_cpu, image_tensor)
					ssim_val = ssim_batch(output_cpu, image_tensor)
				except Exception as exc:
					self._log(f"Metric computation skipped: {exc}")
			else:
				self._log("Metric computation skipped: skimage not available.")

			self.root.after(0, self._update_ui_after_denoise, noise_pct, strength_pct, psnr_val, ssim_val)
		except Exception as exc:
			self.root.after(0, self._handle_denoise_error, exc)

	def _update_ui_after_denoise(self, noise_pct: float, strength_pct: float, psnr_val, ssim_val) -> None:
		if self.output_image_path:
			self._show_image(self.output_image_path, self.output_image_label, is_input=False)
		if self.heatmap_image_path and os.path.exists(self.heatmap_image_path):
			self._show_image(self.heatmap_image_path, self.heatmap_image_label, is_input=False)

		model_cfg = self.cfg.get("model", {}) if self.cfg else {}
		if self.output_image_path:
			self._log(f"Output saved: {os.path.basename(self.output_image_path)}")
		self._log(f"Params: base={model_cfg.get('base_channels', 64)}, cnn={model_cfg.get('num_cnn_blocks', 5)}, "
				  f"transformer={model_cfg.get('num_transformer_blocks', 4)}, residual={model_cfg.get('use_residual_learning', True)}")

		metrics_lines = [
			"Estimated results (approx):",
			f"- Estimated noise level: {noise_pct:.2f}%",
			f"- Estimated denoise strength: {strength_pct:.2f}%",
		]
		if psnr_val is not None:
			metrics_lines.append(f"- PSNR (input vs output): {psnr_val:.2f} dB")
		if ssim_val is not None:
			metrics_lines.append(f"- SSIM (input vs output): {ssim_val:.4f}")
		if self.output_image_path:
			metrics_lines.append(f"- Output file: {os.path.basename(self.output_image_path)}")
		if self.comparison_image_path:
			metrics_lines.append(f"- Comparison file: {os.path.basename(self.comparison_image_path)}")
		metrics_text = "\n".join(metrics_lines)
		self._set_metrics_text(metrics_text)

		self.denoise_btn.configure(state=tk.NORMAL)
		self.save_btn.configure(state=tk.NORMAL)

	def _handle_denoise_error(self, exc: Exception) -> None:
		self._log(f"Denoise failed: {exc}")
		self.denoise_btn.configure(state=tk.NORMAL)
		messagebox.showerror("Denoise failed", str(exc))

	def _show_image(self, path: str, label: tk.Label, is_input: bool) -> None:
		try:
			if not path:
				return
			img = Image.open(path).convert("RGB")
			img = ImageOps.contain(img, (self.preview_w, self.preview_h))
			photo = ImageTk.PhotoImage(img)

			label.configure(image=photo)
			self._image_refs[label] = photo

			if is_input:
				self.input_photo = photo
			else:
				self.output_photo = photo
		except Exception as exc:
			self._log(f"Failed to load image preview: {exc}")

	def _log(self, message: str) -> None:
		timestamp = datetime.now().strftime("%H:%M:%S")
		self.log_text.configure(state=tk.NORMAL)
		self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
		self.log_text.configure(state=tk.DISABLED)
		self.log_text.see(tk.END)

	def _set_metrics_text(self, text: str) -> None:
		self.metrics_text.configure(state=tk.NORMAL)
		self.metrics_text.delete("1.0", tk.END)
		self.metrics_text.insert(tk.END, text)
		self.metrics_text.configure(state=tk.DISABLED)

	def _save_comparison_image(self, out_path: str) -> None:
		try:
			if not self.input_image_path or not self.base_output_image_path:
				return
			img_noisy = Image.open(self.input_image_path).convert("RGB")
			img_output = Image.open(self.base_output_image_path).convert("RGB")

			diff = ImageChops.difference(img_noisy, img_output).convert("L")
			def _boost_pixel(p: int) -> int:
				return min(255, int(p * 10))

			boost = diff.point(_boost_pixel)
			boost = ImageOps.autocontrast(boost)
			heatmap = ImageOps.colorize(boost, black="#2b2f6b", white="#ffcf4a")

			images: List[Image.Image] = [img_noisy, img_output, heatmap]
			min_h = min(img.height for img in images)
			resized: List[Image.Image] = [ImageOps.contain(img, (img.width, min_h)) for img in images]

			gap = 20
			total_w = sum(img.width for img in resized) + gap * (len(resized) - 1)
			canvas = Image.new("RGB", (total_w, min_h), color="#f6f8ff")

			x = 0
			for img in resized:
				canvas.paste(img, (x, 0))
				x += img.width + gap

			canvas.save(out_path)
			if self.heatmap_image_path:
				heatmap.save(self.heatmap_image_path)
			self._log(f"Comparison saved: {os.path.basename(out_path)}")
		except Exception as exc:
			self._log(f"Comparison image skipped: {exc}")

	def _infer_tiled(self, image_tensor: torch.Tensor) -> torch.Tensor:
		inf_cfg = self.cfg.get("inference", {}) if self.cfg else {}
		tile_size = int(inf_cfg.get("tile_size", 0) or 0)
		overlap = int(inf_cfg.get("overlap", 0) or 0)
		denoise_strength = float(inf_cfg.get("denoise_strength", 1.0) or 1.0)
		if self.prompt_settings and "denoise_strength" in self.prompt_settings:
			denoise_strength = float(self.prompt_settings["denoise_strength"])

		_, _, h, w = image_tensor.shape
		if tile_size <= 0 or tile_size >= max(h, w):
			self._log(f"Tiling disabled. Using full image (H={h}, W={w}).")
			self._log(f"Denoise strength: {denoise_strength:.2f}")
			return infer_image(self.model, image_tensor, self.device, window_size=8, denoise_strength=denoise_strength)

		step = max(1, tile_size - overlap)
		self._log(f"Tiling enabled. tile={tile_size}, overlap={overlap}, step={step}.")
		self._log(f"Denoise strength: {denoise_strength:.2f}")

		output = torch.zeros_like(image_tensor)
		weight = torch.zeros_like(image_tensor)

		for y in range(0, h, step):
			for x in range(0, w, step):
				y0 = min(y, max(0, h - tile_size))
				x0 = min(x, max(0, w - tile_size))
				y1 = min(y0 + tile_size, h)
				x1 = min(x0 + tile_size, w)

				tile = image_tensor[:, :, y0:y1, x0:x1]
				out_tile = infer_image(self.model, tile, self.device, window_size=8, denoise_strength=denoise_strength)

				output[:, :, y0:y1, x0:x1] += out_tile
				weight[:, :, y0:y1, x0:x1] += 1.0

		return output / weight.clamp_min(1.0)

	def _get_prompt_value(self, key: str, default: float) -> float:
		if not self.prompt_settings:
			return default
			
		val = self.prompt_settings.get(key)
		if val is None:
			return default
		return float(val)

	def _get_prompt_settings(self) -> Optional[Dict[str, float]]:
		if not self.use_prompt_var.get() or not self.prompt_text:
			if self.prompt_status_label:
				self.prompt_status_label.configure(text="Prompt tuning off")
			return None

		prompt = self.prompt_text.get("1.0", tk.END).strip()
		if not prompt:
			if self.prompt_status_label:
				self.prompt_status_label.configure(text="Prompt empty")
			return None

		import re
		strength_pct = 45.0
		range_match = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*%", prompt)
		if range_match:
			low = float(range_match.group(1))
			high = float(range_match.group(2))
			strength_pct = max(0.0, min(100.0, (low + high) / 2.0))
		else:
			single_match = re.search(r"(\d+)\s*%", prompt)
			if single_match:
				strength_pct = max(0.0, min(100.0, float(single_match.group(1))))

		noise_strength = strength_pct / 100.0
		denoise_strength = 1.0 + noise_strength * 0.8

		if "heavy" in prompt.lower():
			denoise_strength += 0.05
		if "do not oversmooth" in prompt.lower() or "dont oversmooth" in prompt.lower():
			denoise_strength = min(denoise_strength, 1.35)

		clarity = 0.12
		if "enhance clarity" in prompt.lower():
			clarity = 0.15
		if "slightly" in prompt.lower():
			clarity = min(clarity, 0.15)

		contrast = 0.1 if "contrast" in prompt.lower() else 0.05
		sharpen = 0.18 if "sharp" in prompt.lower() or "edges" in prompt.lower() else 0.12

		settings = {
			"denoise_strength": max(1.0, min(1.8, denoise_strength)),
			"post_strength": max(0.2, min(0.7, noise_strength)),
			"clarity": max(0.05, min(0.25, clarity)),
			"contrast": max(0.0, min(0.2, contrast)),
			"sharpen": max(0.05, min(0.3, sharpen)),
		}

		if self.prompt_status_label:
			self.prompt_status_label.configure(
				text=f"Strength {settings['denoise_strength']:.2f} | NR {settings['post_strength']:.2f}"
			)
		self._log(
			"Prompt tuning active: "
			f"denoise_strength={settings['denoise_strength']:.2f}, "
			f"post_strength={settings['post_strength']:.2f}, "
			f"clarity={settings['clarity']:.2f}, "
			f"contrast={settings['contrast']:.2f}, "
			f"sharpen={settings['sharpen']:.2f}"
		)
		return settings

	def on_save_as(self) -> None:
		if not self.output_image_path or not os.path.exists(self.output_image_path):
			messagebox.showinfo("No output", "Please run denoise first.")
			return

		initial_dir = os.path.join(PROJECT_ROOT, "results")
		path = filedialog.asksaveasfilename(
			title="Save denoised image",
			defaultextension=".png",
			initialdir=initial_dir,
			filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("All files", "*.*")],
		)
		if not path:
			return

		try:
			shutil.copy2(self.output_image_path, path)
			self._log(f"Saved output as: {os.path.basename(path)}")
		except Exception as exc:
			messagebox.showerror("Save failed", str(exc))


def main() -> None:
	root = tk.Tk()
	app = DenoiseApp(root)
	root.mainloop()


if __name__ == "__main__":
	main()
