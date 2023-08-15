import folder_paths
import comfy.sd
import hashlib


def parse_name(ckpt_name):
    path = ckpt_name
    filename = path.split("/")[-1]
    filename = filename.split(".")[:-1]
    filename = ".".join(filename)
    return filename


def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read the file in chunks to avoid loading the entire file into memory
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


class CheckpointLoaderWithInfo:
    CATEGORY = 'ImageSaverTools'
    RETURN_TYPES = ("MODEL", "CLIP", "VAE","STRING", "STRING",)
    RETURN_NAMES = ("model", "clip", "vae", "name", "sha256sum",)
    FUNCTION = "load_models"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),}}

    def load_models(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        sha256sum = calculate_sha256(ckpt_path)
        name = parse_name(ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))

        new_out = list(out)
        new_out.pop()
        new_out.append(name)
        new_out.append(sha256sum)

        return tuple(new_out)


class ImageSaveWithMetadata:
    def __init__(self):
        # get default output directory
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(s):
        return {
                    "required": {
                        "images": ("IMAGE", ),
                        "filename": ("STRING", {"default": f'%time_%seed', "multiline": False}),
                        "path": ("STRING", {"default": '', "multiline": False}),
                        "extension": (['png', 'jpeg', 'tiff', 'gif'], ),
                        "quality": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1}),
                    },
                    "optional": {
                        "positive": ("STRING",{"default": '', "multiline": True}),
                        "negative": ("STRING",{"default": '', "multiline": True}),
                        "seed": ("SEED",),
                        "modelname": ("STRING",{"default": '', "multiline": False}),
                        "counter": ("INT",{"default": 0, "min": 0, "max": 0xffffffffffffffff }),
                        "time_format": ("STRING", {"default": "%Y-%m-%d-%H%M%S", "multiline": False}),
                    },
                    "hidden": {
                        "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
                    },
                }

    RETURN_TYPES = ()
    FUNCTION = "save_files"

    OUTPUT_NODE = True

    CATEGORY = "WLSH Nodes/IO"

    def save_files(self, images, positive="unknown", negative="unknown", seed={"seed": 0}, modelname="sd", counter=0, filename='', path="",
    time_format="%Y-%m-%d-%H%M%S", extension='png', quality=100, prompt=None, extra_pnginfo=None):
        filename = make_filename(filename, seed, modelname, counter, time_format)
        comment = "Positive Prompt:\n" + positive + "\nNegative Prompt:\n" + negative + "\nModel: " + modelname + "\nSeed: " + str(seed['seed'])
        output_path = os.path.join(self.output_dir,path)
        
        # create missing paths - from WAS Node Suite
        if output_path.strip() != '':
            if not os.path.exists(output_path.strip()):
                print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
                os.makedirs(output_path, exist_ok=True)    
                
        paths = self.save_images(images, output_path,filename,comment, extension, quality, prompt, extra_pnginfo)
        
        return { "ui": { "images": paths } }

    def save_images(self, images, output_path, filename_prefix="ComfyUI", comment="", extension='png', quality=100, prompt=None, extra_pnginfo=None):
        def map_filename(filename):
            prefix_len = len(filename_prefix)
            prefix = filename[:prefix_len + 1]
            try:
                digits = int(filename[prefix_len + 1:].split('_')[0])
            except:
                digits = 0
            return (digits, prefix)
        
        imgCount = 1
        paths = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = PngInfo()
            
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            if(images.size()[0] > 1):
                filename_prefix += "_{:02d}".format(imgCount)

            file = f"{filename_prefix}.{extension}"
            if extension == 'png':
                img.save(os.path.join(output_path, file), comment=comment, pnginfo=metadata, optimize=True)
            elif extension == 'webp':
                img.save(os.path.join(output_path, file), quality=quality)
            elif extension == 'jpeg':
                img.save(os.path.join(output_path, file), quality=quality, comment=comment, optimize=True)
            elif extension == 'tiff':
                img.save(os.path.join(output_path, file), quality=quality, optimize=True)
            else:
                img.save(os.path.join(output_path, file))
            paths.append(file)
            imgCount += 1
        return(paths)


NODE_CLASS_MAPPINGS = {
    "Load Checkpoint w/Info": CheckpointLoaderWithInfo,
}
