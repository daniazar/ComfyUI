import os

def patch_file(path, old_str, new_str, label):
    if not os.path.exists(path):
        print(f"Skipping {label}: file not found at {path}")
        return
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if old_str in content and new_str not in content:
        content = content.replace(old_str, new_str)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Patched {label}")
    else:
        print(f"⚡ {label} already patched or string not found")

def main():
    print("Applying dynamic ComfyUI patches...")
    
    # 1. Patch quant_ops.py to catch broad Exception instead of just ImportError
    patch_file(
        "/app/vendor/ComfyUI/comfy/quant_ops.py",
        "except ImportError as e:",
        "except Exception as e:",
        "quant_ops.py (comfy_kitchen Exception fallback)"
    )
    
    # 2. Patch utils.py to safely add torch globals only if the method exists
    patch_file(
        "/app/vendor/ComfyUI/comfy/utils.py",
        "torch.serialization.add_safe_globals([ModelCheckpoint, scalar, dtype, Float64DType, encode])",
        "if hasattr(torch.serialization, \"add_safe_globals\"):\n        torch.serialization.add_safe_globals([ModelCheckpoint, scalar, dtype, Float64DType, encode])",
        "utils.py (torch.serialization backward compatibility)"
    )
    
    # 3. Patch init_models.sh to bypass the skipping logic if directory is not empty
    old_init = "    if os.path.exists(ldir) and len(os.listdir(ldir)) > 0:\n        print(f\"Skipping {pattern}, {ldir} is not empty\")\n        return\n"
    patch_file(
        "/app/vendor/ComfyUI/init_models.sh",
        old_init,
        "",
        "init_models.sh (force model downloads)"
    )

if __name__ == "__main__":
    main()
