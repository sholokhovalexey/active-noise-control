import math
import numpy as np
import torch
import torch.nn.functional as F


# def crop_and_compensate_delay(
#     audio, audio_size, ir_size, padding="same", delay_compensation=-1
# ):
#     """Crop audio output from convolution to compensate for group delay.
#     Args:
#     audio: Audio after convolution. Tensor of shape [batch, time_steps].
#     audio_size: Initial size of the audio before convolution.
#     ir_size: Size of the convolving impulse response.
#     padding: Either 'valid' or 'same'. For 'same' the final output to be the
#       same size as the input audio (audio_timesteps). For 'valid' the audio is
#       extended to include the tail of the impulse response (audio_timesteps +
#       ir_timesteps - 1).
#     delay_compensation: Samples to crop from start of output audio to compensate
#       for group delay of the impulse response. If delay_compensation < 0 it
#       defaults to automatically calculating a constant group delay of the
#       windowed linear phase filter from frequency_impulse_response().
#     Returns:
#     Tensor of cropped and shifted audio.
#     Raises:
#     ValueError: If padding is not either 'valid' or 'same'.
#     """
#     # Crop the output.
#     if padding == "valid":
#         crop_size = ir_size + audio_size - 1
#     elif padding == "same":
#         crop_size = audio_size
#     else:
#         raise ValueError(
#             "Padding must be 'valid' or 'same', instead " "of {}.".format(padding)
#         )

#     # Compensate for the group delay of the filter by trimming the front.
#     # For an impulse response produced by frequency_impulse_response(),
#     # the group delay is constant because the filter is linear phase.
#     total_size = int(audio.shape[-1])
#     crop = total_size - crop_size
#     start = ir_size // 2 if delay_compensation < 0 else delay_compensation
#     end = crop - start
#     return audio[:, start:-end]


# def apply_window_to_impulse_response(
#     impulse_response,  # B, n_frames, 2*(n_mag-1)
#     window_size: int = 0,
#     causal: bool = False,
# ):
#     """Apply a window to an impulse response and put in causal form.
#     Args:
#         impulse_response: A series of impulse responses frames to window, of shape
#         [batch, n_frames, ir_size]. ---------> ir_size means size of filter_bank ??????

#         window_size: Size of the window to apply in the time domain. If window_size
#         is less than 1, it defaults to the impulse_response size.
#         causal: Impulse response input is in causal form (peak in the middle).
#     Returns:
#         impulse_response: Windowed impulse response in causal form, with last
#         dimension cropped to window_size if window_size is greater than 0 and less
#         than ir_size.
#     """

#     # If IR is in causal form, put it in zero-phase form.
#     if causal:
#         impulse_response = torch.fftshift(impulse_response, axes=-1)

#     # Get a window for better time/frequency resolution than rectangular.
#     # Window defaults to IR size, cannot be bigger.
#     ir_size = int(impulse_response.size(-1))
#     if (window_size <= 0) or (window_size > ir_size):
#         window_size = ir_size
#     window = nn.Parameter(torch.hann_window(window_size), requires_grad=False).to(
#         impulse_response
#     )

#     # Zero pad the window and put in in zero-phase form.
#     padding = ir_size - window_size
#     if padding > 0:
#         half_idx = (window_size + 1) // 2
#         window = torch.cat(
#             [window[half_idx:], torch.zeros([padding]), window[:half_idx]], axis=0
#         )
#     else:
#         window = window.roll(window.size(-1) // 2, -1)

#     # Apply the window, to get new IR (both in zero-phase form).
#     window = window.unsqueeze(0)
#     impulse_response = impulse_response * window

#     # Put IR in causal form and trim zero padding.
#     if padding > 0:
#         first_half_start = (ir_size - (half_idx - 1)) + 1
#         second_half_end = half_idx + 1
#         impulse_response = torch.cat(
#             [
#                 impulse_response[..., first_half_start:],
#                 impulse_response[..., :second_half_end],
#             ],
#             dim=-1,
#         )
#     else:
#         impulse_response = impulse_response.roll(impulse_response.size(-1) // 2, -1)

#     return impulse_response


# def apply_dynamic_window_to_impulse_response(
#     impulse_response, half_width_frames  # B, n_frames, 2*(n_mag-1) or 2*n_mag-1
# ):  # B，n_frames, 1
#     ir_size = int(impulse_response.size(-1))  # 2*(n_mag -1) or 2*n_mag-1

#     window = (
#         torch.arange(-(ir_size // 2), (ir_size + 1) // 2).to(impulse_response)
#         / half_width_frames
#     )
#     window[window > 1] = 0
#     window = (
#         1 + torch.cos(np.pi * window)
#     ) / 2  # B, n_frames, 2*(n_mag -1) or 2*n_mag-1

#     impulse_response = impulse_response.roll(ir_size // 2, -1)
#     impulse_response = impulse_response * window
#     return impulse_response
