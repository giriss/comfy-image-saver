# Save image with generation metadata on ComfyUI

All the tools you need to save images with their **generation metadata** on ComfyUI. Compatible with *Civitai* & *Prompthero* geninfo auto-detection. Works with `png`, `jpeg` and `webp`.

You can find the example workflow file named `example-workflow.json`.

![example-workflow](https://github.com/giriss/comfy-image-saver/assets/2811408/e231237b-f91a-4679-b3ae-2618080c8e39)

## How to install?

### Method 1: Easiest (Recommended)
If you have *ComfyUI-Manager*, you can simply search "**Save Image with Generation Metadata**" and install these custom nodes 🎉


### Method 2: Easy
If you don't have *ComfyUI-Manager*, then:
- Using CLI, go to the ComfyUI folder
- `cd custom_nodes`
- `git clone git@github.com:giriss/comfy-image-saver.git`
- `cd comfy-image-saver`
- `pip install -r requirements.txt`
- Start/restart ComfyUI 🎉

## Autodetection in action

![Screenshot 2023-08-17 at 13 15 18](https://github.com/giriss/comfy-image-saver/assets/2811408/785f2475-8f9a-45c9-9d38-855161a98495)

## Customization of file/folder names

You can use following placeholders:

- `%date`
- `%seed`
- `%counter`
- `%sampler_name`
- `%steps`
- `%cfg`
- `%scheduler`
- `%time` *– format taken from `time_format`*
- `%model` *– full name of model file*
- `%basemodelname` *– name of model (without file extension)*

Example:

| `filename` value | Result file name |
| --- | --- |
| `%time-%basemodelname-%cfg-%steps-%sampler_name-%scheduler-%seed` | `2023-11-16-131331-Anything-v4.5-pruned-mergedVae-7.0-25-dpm_2-normal-1_01.png` |
