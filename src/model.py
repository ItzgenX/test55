from abc import ABC, abstractmethod
from typing import Union, Literal
import torch
from torch import nn
from src.utils import DataProvider
import src.lora as loras
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers import AutoencoderTiny, AutoencoderKL
from src.utils import getattr_recursive

import torch.nn.functional as F
from pydoc import locate
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)

from tqdm.auto import tqdm
import random

from diffusers import ControlNetModel
from torchvision.transforms import Compose
from typing import Callable

ATTENTION_MODULES = ["to_k", "to_v"]

# only for SD15
CONV_MODULES = ["conv1", "conv2"]

ADAPTION_MODE = Literal[
    "full_attention",
    "only_self",
    "only_cross",
    "only_conv",
    "only_first_conv",
    "only_res_conv",
    "full",
    "no_cross",
    "only_value",
    # below only works for sdxl
    "b-lora_style",
    "b-lora_content",
    "b-lora",
    "sdxl_cross",
    "sdxl_self",
    "sdxl_inner",
]

CONDITION_MODE = Literal["style", "structure"]


class ModelBase(ABC, nn.Module):

    def __init__(
        self,
        pipeline_type: str,
        model_name: str,
        local_files_only: bool = True,
        c_dropout: float = 0.05,
        guidance_scale: float = 7.5,
        use_controlnet: bool = False,
        annotator: None | nn.Module = None,
        tiny_vae: bool = False,
        vae_path: str | None = None,
    ) -> None:
        super().__init__()
        self.params_to_optimize: list[nn.Parameter] = []
        self.lora_state_dict_keys: dict[str, list[str]] = {}
        self.lora_layers: dict[str, list[nn.Module]] = {}
        self.lora_transforms: list[Compose | None] = []

        self.encoders: list[nn.Module] = list()
        self.mappers: list[nn.Module] = list()
        self.dps: list[DataProvider] = []

        self.tiny_vae = tiny_vae
        self.c_dropout = c_dropout
        self.guidance_scale = guidance_scale
        self.use_controlnet = use_controlnet

        addition_config = {}

        # Note that this requires the controlnet pipe which also has to be set in the config

        assert not (tiny_vae and vae_path), "tiny_vae and vae_path are mutually exclusive"

        if tiny_vae:
            vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", local_files_only=local_files_only)
            addition_config["vae"] = vae

        if vae_path:
            # swap in a better-trained VAE than whatever ships with the base checkpoint
            # e.g. "madebyollin/sdxl-vae-fp16-fix" for SDXL, "stabilityai/sd-vae-ft-mse" for SD1.5
            vae = AutoencoderKL.from_pretrained(vae_path, local_files_only=local_files_only)
            addition_config["vae"] = vae

        if self.use_controlnet:
            assert annotator is not None, "Need annotator for controlnet"

            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-depth",
                use_safetensors=True,
                local_files_only=local_files_only,
                **addition_config,
            )
            # controlnet.requires_grad_(False)
            # controlnet.eval()
            addition_config["controlnet"] = controlnet

            # fix this cheap work around!
            self.encoders.append(annotator)
            self.mappers.append(controlnet)

        self.pipe: DiffusionPipeline = locate(pipeline_type).from_pretrained(
            model_name,
            local_files_only=local_files_only,
            safety_checker=None,  # too anoying
            safe_tensors=True,
            **addition_config,
        )
        assert isinstance(self.pipe, DiffusionPipeline)

        # Base checkpoints are not guaranteed internally dtype-consistent --
        # e.g. stabilityai/stable-diffusion-xl-base-1.0's default (non-"fp16"-
        # variant) files ship both text encoders in fp16 while UNet/VAE are
        # fp32, since no torch_dtype was requested at download time. Our own
        # forward() builds every tensor by hand so it never hit this, but
        # sample()'s call into the stock pipeline infers a working dtype from
        # the (fp16) text embeddings and casts latents to match, feeding fp16
        # latents into the fp32 UNet -- "mat1 and mat2 must have the same
        # dtype". Standardize once, here, instead of per call-site.
        self.pipe.to(dtype=torch.float32)

        self.noise_scheduler = DDPMScheduler.from_config(
            {**self.pipe.scheduler.config, "rescale_betas_zero_snr": False},
            subfolder="scheduler",
        )

        self.max_depth = len(self.pipe.unet.config["block_out_channels"]) - 1

        # we register all the individual pipeline modules here
        # such that all the typical calls like .to and .prepare effect them.
        self.unet = self.pipe.unet
        self.unet.requires_grad_(False)

        self.vae = self.pipe.vae
        self.vae.requires_grad_(False)
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer

        self.text_encoder.requires_grad_(False)

        # handle sdxl case
        if hasattr(self.pipe, "text_encoder_2"):
            self.text_encoder_2 = self.pipe.text_encoder_2
            self.text_encoder_2.requires_grad_(False)

    def add_lora_to_unet(
        self,
        config: dict,
        name: str,
        data_provider: DataProvider,
        encoder: nn.Module,
        mapper: nn.Module,
        optimize: bool = True,
        transforms: list[Callable] = [],
    ):
        self.rank = config.rank
        self.c_dim = config.c_dim
        unet = self.unet
        sd = unet.state_dict()

        self.mappers.append(mapper)
        self.encoders.append(encoder)
        self.dps.append(data_provider)

        self.lora_transforms.append(Compose(transforms) if len(transforms) > 0 else None)

        print(f"adding {len(transforms)} transforms to LoRA {name}")

        lora_cls = config.lora_cls
        adaption_mode = config.adaption_mode

        if not optimize:
            mapper.eval()
            mapper.requires_grad_(False)

        local_lora_sd_keys: list[str] = []

        for path, w in sd.items():
            class_config = {**config}
            del class_config["lora_cls"]
            del class_config["adaption_mode"]

            _continue = True
            if adaption_mode == "full_attention" and "attn" in path:
                _continue = False

            if adaption_mode == "only_self" and "attn1" in path:
                _continue = False

            if adaption_mode == "only_cross" and "attn2" in path:
                _continue = False

            if adaption_mode == "only_conv" and ("conv1" in path or "conv2" in path):
                _continue = False

            # only the first conv layer in each resnet block
            if adaption_mode == "only_first_conv" and "0.conv1" in path:
                _continue = False

            if adaption_mode == "only_res_conv" and ("0.conv1" in path or "1.conv1" in path):
                _continue = False

            if adaption_mode == "full" and ("attn" in path or "conv" in path):
                _continue = False

            if adaption_mode == "no_cross" and "attn2" not in path:
                _continue = False

            if adaption_mode == "b-lora_content" and ("up_blocks.0.attentions.0" in path and "attn" in path):
                _continue = False

            if adaption_mode == "b-lora_style" and ("up_blocks.0.attentions.1" in path and "attn" in path):
                _continue = False

            if (
                adaption_mode == "b-lora"
                and ("up_blocks.0.attentions.0" in path or "up_blocks.0.attentions.1" in path)
                and "attn" in path
            ):
                _continue = False

                # supposed setting content to have no effect
                # if "up_blocks.0.attentions.0" in path:
                # class_config["lora_scale"] = 0.0

            # "down_blocks.2.attentions.1" in path or
            if (
                adaption_mode == "sdxl_inner"
                and ("mid_block" in path or "up_blocks.0.attentions.0" in path or "up_blocks.0.attentions.1" in path)
                and "attn2" in path
            ):
                _continue = False

            if (
                adaption_mode == "sdxl_cross"
                and ("down_blocks.2" in path or "up_blocks.0" in path or "mid_block" in path)
                and "attn2" in path
            ):
                _continue = False

            if (
                adaption_mode == "sdxl_self"
                and ("down_blocks.2" in path or "up_blocks.0" in path or "mid_block" in path)
                and "attn1" in path
            ):
                _continue = False

            if _continue:
                continue

            if "bias" in path:
                # we handle the bias together with the weight
                # this is only relevant for the conv layers
                continue

            parent_path = ".".join(path.split(".")[:-2])
            target_path = ".".join(path.split(".")[:-1])
            target_name = path.split(".")[-2]
            parent_module = getattr_recursive(unet, parent_path)
            target_module = getattr_recursive(unet, target_path)

            if "mid_block" in path:
                depth = self.max_depth
            elif "down_blocks" in path:
                depth = int(path.split("down_blocks.")[1][0])
            elif "up_blocks" in path:
                depth = self.max_depth - int(path.split("up_blocks.")[1][0])
            else:
                raise ValueError(f"Unknown module {path}")

            lora = None
            if "attn" in path:
                if not any([m in path for m in ATTENTION_MODULES]):
                    continue

                lora = getattr(loras, lora_cls)(
                    out_features=target_module.out_features,
                    in_features=target_module.in_features,
                    data_provider=data_provider,
                    depth=depth,
                    **class_config,
                )

                # W is the original weight matrix
                # those layers have no bias
                lora.W.load_state_dict({path.split(".")[-1]: w})

                if lora_cls == "IPLinear":
                    # for faster convergence
                    lora.W_IP.load_state_dict({path.split(".")[-1]: w})

            if "conv" in path:
                has_bias = target_module.bias is not None
                lora = getattr(loras, lora_cls)(
                    in_channels=target_module.in_channels,
                    out_channels=target_module.out_channels,
                    kernel_size=target_module.kernel_size,
                    stride=target_module.stride,
                    padding=target_module.padding,
                    data_provider=data_provider,
                    depth=depth,
                    bias=has_bias,
                    **class_config,
                )

                # W mirrors the original conv exactly, bias included -- only
                # look up/load a bias term when the original actually has one
                # (has_bias above already made W biasless otherwise, so there
                # is nothing to load and no state-dict key to expect)
                if has_bias:
                    bias_path = ".".join(path.split(".")[:-1] + ["bias"])
                    if bias_path not in sd:
                        raise ValueError(
                            f"Conv layer '{path}' has bias=True but no matching bias entry "
                            f"'{bias_path}' was found in the state dict -- state dict may not "
                            f"match this UNet's actual architecture."
                        )
                    lora.W.load_state_dict({path.split(".")[-1]: w, "bias": sd[bias_path]})
                else:
                    lora.W.load_state_dict({path.split(".")[-1]: w})

            if lora is None:
                raise ValueError(f"Unknown module {path}")

            for k in lora.state_dict().keys():
                # W is by design the original weight matrix which we don't need to save
                if k.split(".")[0] == "W":
                    continue

                local_lora_sd_keys.append(f"{target_path}.{k}")

            self.lora_state_dict_keys[name] = local_lora_sd_keys

            setattr(
                parent_module,
                target_name,
                lora,
            )

            if optimize:
                for p in lora.parameters():
                    if p.requires_grad:
                        self.params_to_optimize.append(p)
            else:
                lora.eval()
                for p in lora.parameters():
                    p.requires_grad_(False)

            self.lora_layers[name] = [lora] + self.lora_layers.get(name, [])

    def get_lora_state_dict(self, unet: Union[nn.Module, None] = None):
        lora_sd = {}

        if unet is None:
            unet = self.unet

        for k, v in unet.state_dict().items():
            for n, keys in self.lora_state_dict_keys.items():
                if n not in lora_sd:
                    lora_sd[n] = {}

                if k in keys:
                    lora_sd[n][k] = v.cpu()

        return lora_sd

    def soften_conditioning(self, c: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Box-blur a conditioning map so its edges aren't perfectly sharp.

        A segmentation map's class boundaries are pixel-exact -- no real
        photo has that (a curb, a shadow, a shoreline all blend a little).
        Feeding the model perfectly crisp boundaries every time risks it
        learning to render unnaturally crisp, cartoonish edges. kernel_size
        <= 1 is a no-op. Applied once, before generation, to the raw
        conditioning tensor -- not a per-step thing.
        """
        if kernel_size is None or kernel_size <= 1:
            return c
        if kernel_size % 2 == 0:
            kernel_size += 1  # keep it odd so padding is symmetric

        channels = c.shape[1]
        weight = torch.ones(channels, 1, kernel_size, kernel_size, device=c.device, dtype=c.dtype)
        weight = weight / (kernel_size**2)
        return F.conv2d(c, weight, padding=kernel_size // 2, groups=channels)

    def lora_scale_schedule_callback(
        self,
        lora_scale_start: float,
        lora_scale_end: float,
        lora_scale_decay_start_frac: float,
        num_inference_steps: int,
    ):
        """Build a diffusers callback_on_step_end that smoothly varies every
        patched LoRA layer's .lora_scale across the denoising trajectory --
        full strength (lora_scale_start) while the model is still deciding
        the overall layout, decaying to lora_scale_end once the layout is
        settled (after lora_scale_decay_start_frac of the steps have run),
        so late steps can lean more on the base model's own texture/detail
        prior instead of a flat, rigid conditioning map.

        Reuses the exact same per-layer .lora_scale attribute toggle_loras()
        already flips between train/val (src/utils.py) -- just varied
        smoothly per-step instead of once per run. Returns None (no callback
        needed) when start == end, so passing the defaults is a true no-op.
        """
        if lora_scale_start == lora_scale_end:
            return None

        all_layers = [layer for layers in self.lora_layers.values() for layer in layers]

        def _callback(pipe, step_index, timestep, callback_kwargs):
            frac = step_index / max(num_inference_steps - 1, 1)
            if frac <= lora_scale_decay_start_frac:
                scale = lora_scale_start
            else:
                t = (frac - lora_scale_decay_start_frac) / max(1.0 - lora_scale_decay_start_frac, 1e-6)
                scale = lora_scale_start + t * (lora_scale_end - lora_scale_start)
            for layer in all_layers:
                layer.lora_scale = scale
            return callback_kwargs

        return _callback

    @abstractmethod
    def get_input(self, imgs: torch.Tensor, prompts: list[str]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raise NotImplementedError()

    # -> epsilon, loss, x0
    def forward_easy(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self(args, kwargs)

    def sample(self, *args, **kwargs):
        return self.pipe(*args, **kwargs).images


class SD15(ModelBase):
    def __init__(self, pipeline_type, model_name, *args, **kwargs) -> None:
        super().__init__(pipeline_type, model_name, *args, **kwargs)

    @torch.no_grad()
    def get_input(self, imgs: torch.Tensor, prompts: list[str]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert len(imgs.shape) == 4
        assert imgs.min() >= -1.0
        assert imgs.max() <= 1.0

        imgs = imgs.clip(-1.0, 1.0)

        # Convert images to latent space
        if self.tiny_vae:
            latents = self.vae.encode(imgs).latents
        else:
            latents = self.vae.encode(imgs).latent_dist.sample()

        latents = latents * self.vae.config.scaling_factor

        # prompt dropout
        prompts = ["" if random.random() < self.c_dropout else p for p in prompts]

        # prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
        #     prompt=prompts,
        #     device=self.unet.device,
        #     num_images_per_prompt=1,
        #     do_classifier_free_guidance=False,
        # )

        # do it manually to avoid stupid warnings
        input_ids = self.tokenizer(
            prompts,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        prompt_embeds = self.text_encoder(input_ids.to(imgs.device))["last_hidden_state"]

        # assert (prompt_embeds - prompt_embeds).mean() < 1e-6
        # assert (prompt_embeds == prompt_embeds).all()

        c = {
            "prompt_embeds": prompt_embeds,
        }

        return latents, c

    def forward(
        self,
        latents: torch.Tensor,
        c: dict[str, torch.Tensor],
        cs: list[torch.Tensor],
        timesteps: torch.Tensor,
        noise: torch.Tensor,
        cfg_mask: list[bool] | None = None,
        skip_encode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        prompt_embeds = c["prompt_embeds"]
        bsz = latents.shape[0]
        encoders = self.encoders
        mappers = self.mappers

        additional_inputs = {}
        if self.use_controlnet:
            # controlnet related stuff is always at index 0
            cn_input = cs[0]
            cs = cs[1:]

            controlnet = mappers[0]
            mappers = mappers[1:]

            annotator = encoders[0]
            encoders = encoders[1:]

            with torch.no_grad():
                cn_cond = annotator(cn_input)

            down_block_res_samples, mid_block_res_sample = controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=cn_cond,
                conditioning_scale=1.0,
                return_dict=False,
            )

            additional_inputs["down_block_additional_residuals"] = down_block_res_samples
            additional_inputs["mid_block_additional_residual"] = mid_block_res_sample

        # add our lora conditioning
        # cs in [-1, 1]
        for i, (encoder, dp, mapper, lora_c) in enumerate(zip(encoders, self.dps, mappers, cs)):
            if cfg_mask is None or cfg_mask[i]:
                dropout_mask = torch.rand(bsz, device=lora_c.device) < self.c_dropout

                # apply dropout for cfg -- clone first: cs entries can alias the
                # same tensor across multiple LoRA slots (e.g. cs = [seg] * n_loras),
                # so an in-place write here would leak this LoRA's dropout mask
                # into every other slot sharing the same underlying storage.
                lora_c = lora_c.clone()
                lora_c[dropout_mask] = torch.zeros_like(lora_c[dropout_mask])

            if skip_encode:
                cond = lora_c
            else:
                # some encoders we want to finetune
                # so no torch.no_grad() here
                # instead we set requires_grad in the corresponding classes/configs
                t = self.lora_transforms[i]
                if t is not None:
                    lora_c = t(lora_c)
                cond = encoder(lora_c)
            mapped_cond = mapper(cond)
            dp.set_batch(mapped_cond)

        # Predict the noise residual
        model_pred = self.unet(
            noisy_latents, timesteps, encoder_hidden_states=prompt_embeds, **additional_inputs
        ).sample

        # get x0 prediction
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=model_pred.device, dtype=model_pred.dtype)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(model_pred.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(model_pred.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        x0 = (noisy_latents - sqrt_one_minus_alpha_prod * model_pred) / sqrt_alpha_prod

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        return model_pred, loss, x0, cond

    def forward_easy(
        self,
        imgs: torch.Tensor,
        prompts: list[str],
        cs: list[torch.Tensor],
        cfg_mask: list[bool] | None = None,
        skip_encode: bool = False,
        batch: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        latents, c = self.get_input(imgs, prompts)

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        # timesteps = timesteps.long()

        return self(
            latents=latents,
            c=c,
            cs=cs,
            timesteps=timesteps,
            noise=noise,
            cfg_mask=cfg_mask,
            skip_encode=skip_encode,
        )

    @torch.no_grad()
    def sample_custom(
        self,
        prompt,
        num_images_per_prompt,
        cs: list[torch.Tensor],
        generator,
        cfg_mask: list[bool] | None = None,
        prompt_offset_step: int = 0,
        skip_encode: bool = False,
        **kwargs,
    ):
        height = self.unet.config.sample_size * self.pipe.vae_scale_factor
        width = self.unet.config.sample_size * self.pipe.vae_scale_factor

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)

        batch_size = batch_size * num_images_per_prompt

        device = self.unet.device

        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt, device, num_images_per_prompt, True
        )  # do cfg
        dtype = prompt_embeds.dtype

        # for cfg
        c_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds]).to(dtype)
        uc_prompt_embeds = torch.cat([negative_prompt_embeds, negative_prompt_embeds]).to(dtype)

        # we have to do two separate forward passes for the cfg with the loras
        # add our lora conditioning
        for i, (encoder, dp, mapper, c) in enumerate(zip(self.encoders, self.dps, self.mappers, cs)):

            if c.shape[0] != batch_size:
                assert c.shape[0] == 1
                c = torch.cat(batch_size * [c])  # repeat along batch dim

            neg_c = torch.zeros_like(c)
            if cfg_mask is not None and not cfg_mask[i]:
                print("no cfg for lora nr", i)
                c = torch.cat([c, c])
            else:
                c = torch.cat([neg_c, c])

            if skip_encode:
                cond = c
            else:
                cond = encoder(c)
            mapped_cond = mapper(cond)
            if isinstance(mapped_cond, tuple) or isinstance(mapped_cond, list):
                mapped_cond = [mc.to(dtype) for mc in mapped_cond]
            else:
                mapped_cond = mapped_cond.to(dtype)

            dp.set_batch(mapped_cond)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.pipe.scheduler, 50, device)

        # 5. Prepare latent variables
        num_channels_latents = 4  # self.unet.config.in_channels
        latents = self.pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            c_prompt_embeds.dtype,
            device,
            generator,
        )

        for i, t in tqdm(enumerate(timesteps)):
            # cfg
            latent_model_input = torch.cat([latents] * 2)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=(c_prompt_embeds if i >= prompt_offset_step else uc_prompt_embeds),
                return_dict=False,
            )[0]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        latents = latents.to(torch.float32)
        image = self.vae.decode(
            latents / self.vae.config.scaling_factor,
            return_dict=False,
            generator=generator,
        )[0]
        do_denormalize = [True] * image.shape[0]

        image = self.pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=do_denormalize)

        return image

        # with self.progress_bar(total=num_inference_steps) as progress_bar:

    @torch.no_grad()
    def sample(self, *args, **kwargs):
        return self.sample_easy(*args, **kwargs)

    @torch.no_grad()
    def sample_easy(
        self,
        prompt,
        num_images_per_prompt,
        cs: list[torch.Tensor],
        generator,
        cfg_mask: list[bool] | None = None,
        prompt_offset_step: int = 0,
        skip_encode: bool = False,
        # dtype=torch.float32,
        **kwargs,
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)

        batch_size = batch_size * num_images_per_prompt

        mappers = self.mappers
        encoders = self.encoders
        if self.use_controlnet:
            # controlnet related stuff is always at index 0
            cn_input = cs[0]
            cs = cs[1:]

            mappers = mappers[1:]

            annotator = encoders[0]
            encoders = encoders[1:]

            with torch.no_grad():
                cn_cond = annotator(cn_input)

            kwargs["image"] = cn_cond

        # we have to do two separate forward passes for the cfg with the loras
        # add our lora conditioning
        for i, (encoder, dp, mapper, c) in enumerate(zip(encoders, self.dps, mappers, cs)):

            if c.shape[0] != batch_size:
                assert c.shape[0] == 1
                c = torch.cat(batch_size * [c])  # repeat along batch dim

            neg_c = torch.zeros_like(c)
            if cfg_mask is not None and not cfg_mask[i]:
                print("no cfg for lora nr", i)
                c = torch.cat([c, c])
            else:
                c = torch.cat([neg_c, c])

            if skip_encode:
                cond = c
            else:
                cond = encoder(c)
            mapped_cond = mapper(cond)
            # if isinstance(mapped_cond, tuple) or isinstance(mapped_cond, list):
            #     mapped_cond = [mc.to(dtype) for mc in mapped_cond]
            # else:
            #     mapped_cond = mapped_cond.to(dtype)

            dp.set_batch(mapped_cond)

        return self.pipe(
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            **kwargs,
        ).images


class SDXL(ModelBase):
    def __init__(self, pipeline_type, model_name, *args, **kwargs) -> None:
        super().__init__(pipeline_type, model_name, *args, **kwargs)

    def get_input(self, imgs: torch.Tensor, prompts: list[str]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raise NotImplementedError()

    def compute_time_ids(self, device, weight_dtype, size: tuple[int, int] = (1024, 1024)):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        #
        # size = (height, width) of the actual training image, not a fixed
        # placeholder -- SDXL's micro-conditioning learns resolution/crop
        # -aware behavior from this signal, so training must tell it the
        # truth. Sampling (diffusers' own pipeline __call__) already derives
        # this correctly from the real height/width kwargs; this path is the
        # one that used to hardcode (1024, 1024) regardless of dataset size,
        # creating a train/inference mismatch for any non-1024-square dataset.
        original_size = size
        target_size = size
        crops_coords_top_left = (0, 0)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(device, dtype=weight_dtype)
        return add_time_ids

    def get_conditioning(
        self,
        prompts: list[str],
        bsz: int,
        device: torch.device,
        dtype: torch.dtype,
        do_cfg=False,
        size: tuple[int, int] = (1024, 1024),
    ):
        add_time_ids = self.compute_time_ids(device, dtype, size)
        negative_add_time_ids = add_time_ids  # no conditioning for now

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompts,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
        )

        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            # repeat each half to bsz rows *before* concatenating, so the row
            # order (all-negative rows, then all-positive rows) matches
            # prompt_embeds/add_text_embeds above -- repeating the
            # pre-concatenated [2, 6] tensor instead would tile it as
            # [neg,pos,neg,pos,...], misaligning every row against its
            # prompt/pooled-embed counterpart for any bsz > 1.
            add_time_ids = torch.cat(
                [negative_add_time_ids.repeat(bsz, 1), add_time_ids.repeat(bsz, 1)], dim=0
            )
        else:
            # prompt_embeds = prompt_embeds
            add_text_embeds = pooled_prompt_embeds
            add_time_ids = add_time_ids.repeat(bsz, 1)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device)

        return {
            "prompt_embeds": prompt_embeds,
            "add_text_embeds": add_text_embeds,
            "add_time_ids": add_time_ids,
        }

    def forward_easy(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        imgs: torch.Tensor,
        prompts: list[str],
        cs: list[torch.Tensor],
        cfg_mask: list[bool] | None = None,
        skip_encode: bool = False,
        batch: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(imgs.shape) == 4
        assert imgs.min() >= -1.0
        assert imgs.max() <= 1.0

        B = imgs.shape[0]

        with torch.no_grad():
            # Convert images to latent space
            imgs = imgs.to(self.unet.device)
            latents = self.pipe.vae.encode(imgs).latent_dist.sample()
            latents = latents * self.pipe.vae.config.scaling_factor

            # prompt dropout
            prompts = ["" if random.random() < self.c_dropout else p for p in prompts]

            c = self.get_conditioning(
                prompts, B, latents.device, latents.dtype, size=(imgs.shape[-2], imgs.shape[-1])
            )

        unet_added_conditions = {
            "time_ids": c["add_time_ids"],
            "text_embeds": c["add_text_embeds"],
        }
        prompt_embeds_input = c["prompt_embeds"]

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=latents.device,
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        encoders = self.encoders
        mappers = self.mappers

        # controlnet related stuff is always at index 0 -- must be sliced off
        # before the main lora loop below, matching SD15.forward(); this class
        # previously zipped self.encoders/self.mappers/cs against self.dps
        # directly, which never has a controlnet entry, silently misaligning
        # every real lora's (encoder, mapper, cs) against the wrong self.dps
        # slot whenever use_controlnet=True, and never actually running the
        # controlnet forward pass at all.
        additional_inputs = {}
        if self.use_controlnet:
            cn_input = cs[0]
            cs = cs[1:]

            controlnet = mappers[0]
            mappers = mappers[1:]

            annotator = encoders[0]
            encoders = encoders[1:]

            with torch.no_grad():
                cn_cond = annotator(cn_input)

            down_block_res_samples, mid_block_res_sample = controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds_input,
                controlnet_cond=cn_cond,
                conditioning_scale=1.0,
                return_dict=False,
            )

            additional_inputs["down_block_additional_residuals"] = down_block_res_samples
            additional_inputs["mid_block_additional_residual"] = mid_block_res_sample

        # add our lora conditioning
        for i, (encoder, dp, mapper, c) in enumerate(zip(encoders, self.dps, mappers, cs)):
            if cfg_mask is None or cfg_mask[i]:
                dropout_mask = torch.rand(bsz, device=c.device) < self.c_dropout

                # apply dropout for cfg -- clone first: cs entries can alias the
                # same tensor across multiple LoRA slots (e.g. cs = [seg] * n_loras),
                # so an in-place write here would leak this LoRA's dropout mask
                # into every other slot sharing the same underlying storage.
                c = c.clone()
                c[dropout_mask] = torch.zeros_like(c[dropout_mask])

            if skip_encode:
                cond = c
            else:
                with torch.no_grad():
                    t = self.lora_transforms[i]
                    if t is not None:
                        c = t(c)
                    cond = encoder(c)
            mapped_cond = mapper(cond)
            dp.set_batch(mapped_cond)

        # Predict the noise residual
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            prompt_embeds_input,
            added_cond_kwargs=unet_added_conditions,
            **additional_inputs,
        ).sample

        # get the x0 prediction in ddpm sampling
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=model_pred.device, dtype=model_pred.dtype)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(model_pred.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(model_pred.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        x0 = (noisy_latents - sqrt_one_minus_alpha_prod * model_pred) / sqrt_alpha_prod

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise

        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        return model_pred, loss, x0, cond

    @torch.no_grad()
    def sample(
        self,
        prompt,
        num_images_per_prompt,
        cs: list[torch.Tensor],
        generator,
        cfg_mask: list[bool] | None = None,
        prompt_offset_step: int = 0,
        skip_encode: bool = False,
        dtype=torch.float32,
        batch: dict | None = None,
        conditioning_kernel_size: int = 0,
        lora_scale_start: float = 1.0,
        lora_scale_end: float = 1.0,
        lora_scale_decay_start_frac: float = 0.3,
        **kwargs,
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)

        batch_size = batch_size * num_images_per_prompt

        device = self.unet.device

        prompt_embeds = None
        pooled_prompt_embeds = None

        # soften hard conditioning-map edges once, before CFG-doubling/encoding
        cs = [self.soften_conditioning(c, conditioning_kernel_size) for c in cs]

        # Resolve once, use consistently below -- a caller-supplied guidance_scale
        # (e.g. segformer_inference.py's cfg.inference.guidance_scale) must drive
        # BOTH the per-LoRA CFG-doubling decision and the actual pipe call, or the
        # two could disagree. Popped (not just read) so it isn't also forwarded
        # via **kwargs, which would collide with the explicit kwarg below.
        guidance_scale = kwargs.pop("guidance_scale", self.guidance_scale)

        callback = self.lora_scale_schedule_callback(
            lora_scale_start, lora_scale_end, lora_scale_decay_start_frac, kwargs.get("num_inference_steps", 50)
        )
        if callback is not None:
            kwargs["callback_on_step_end"] = callback

        mappers = self.mappers
        encoders = self.encoders
        if self.use_controlnet:
            # controlnet related stuff is always at index 0 -- must be sliced
            # off before the main lora loop below, matching SD15.sample_easy();
            # this method previously zipped self.encoders/self.mappers/cs
            # against self.dps directly, which never has a controlnet entry.
            cn_input = cs[0]
            cs = cs[1:]

            mappers = mappers[1:]

            annotator = encoders[0]
            encoders = encoders[1:]

            with torch.no_grad():
                cn_cond = annotator(cn_input)

            kwargs["image"] = cn_cond

        # we have to do two separate forward passes for the cfg with the loras
        # add our lora conditioning
        for i, (encoder, dp, mapper, c) in enumerate(zip(encoders, self.dps, mappers, cs)):

            if c.shape[0] != batch_size:
                assert c.shape[0] == 1
                c = torch.cat(batch_size * [c])  # repeat along batch dim

            neg_c = torch.zeros_like(c)
            if guidance_scale > 1:
                if cfg_mask is not None and not cfg_mask[i]:
                    print("no cfg for lora nr", i)
                    c = torch.cat([c, c])
                else:
                    c = torch.cat([neg_c, c])

            if skip_encode:
                cond = c
            else:
                cond = encoder(c)
            mapped_cond = mapper(cond)
            # if isinstance(mapped_cond, tuple) or isinstance(mapped_cond, list):
            #     mapped_cond = [mc.to(dtype) for mc in mapped_cond]
            # else:
            #     mapped_cond = mapped_cond.to(dtype)

            dp.set_batch(mapped_cond)

        return self.pipe(
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            guidance_scale=guidance_scale,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            **kwargs,
        ).images
