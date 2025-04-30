import math
import numpy as np
import torch

# from common.math import sin, cos, sqrt
# from .utils import crop_and_compensate_delay


def mag(x, eps=1e-8):
    real = x.real**2
    imag = x.imag**2
    mag = torch.sqrt(torch.clamp(real + imag, min=eps))
    return mag


def polyval(p, x):
    """Evaluate a polynomial at specific values.

    Args:
        p (batch, coeffs), [a_{N}, a_{N-1}, ..., a_1, a_0]
        x (values)

    Returns:
        v (batch, values)

    """
    bs, N = p.size()
    val = torch.tensor([0], device=p.device)
    for i in range(N - 1):
        val = (val + p[:, i]) * x.view(-1, 1)
    return (val + p[:, -1]).T


def roots(p):
    n = p.size(-1)
    A = torch.diag(torch.ones(n - 2), -1)
    A[0, :] = -p[1:] / p[0]
    # r = torch.symeig(A)
    r = torch.eig(A)
    # r_eig = np.linalg.eigvals(A)
    # r_np = np.roots(p.cpu().numpy())
    return r



# def get_fft_size(frame_size: int, ir_size: int, power_of_2: bool = True):
#     """Calculate final size for efficient FFT.
#     Args:
#     frame_size: Size of the audio frame.
#     ir_size: Size of the convolving impulse response.
#     power_of_2: Constrain to be a power of 2. If False, allow other 5-smooth
#       numbers. TPU requires power of 2, while GPU is more flexible.
#     Returns:
#     fft_size: Size for efficient FFT.
#     """
#     convolved_frame_size = ir_size + frame_size - 1
#     if power_of_2:
#         # Next power of 2.
#         fft_size = int(2 ** np.ceil(np.log2(convolved_frame_size)))
#     else:
#         fft_size = convolved_frame_size
#     return fft_size


# def fft_convolve(audio, impulse_response):  # B, n_frames, 2*(n_mags-1)
#     """Filter audio with frames of time-varying impulse responses.
#     Time-varying filter. Given audio [batch, n_samples], and a series of impulse
#     responses [batch, n_frames, n_impulse_response], splits the audio into frames,
#     applies filters, and then overlap-and-adds audio back together.
#     Applies non-windowed non-overlapping STFT/ISTFT to efficiently compute
#     convolution for large impulse response sizes.
#     Args:
#         audio: Input audio. Tensor of shape [batch, audio_timesteps].
#         impulse_response: Finite impulse response to convolve. Can either be a 2-D
#         Tensor of shape [batch, ir_size], or a 3-D Tensor of shape [batch,
#         ir_frames, ir_size]. A 2-D tensor will apply a single linear
#         time-invariant filter to the audio. A 3-D Tensor will apply a linear
#         time-varying filter. Automatically chops the audio into equally shaped
#         blocks to match ir_frames.
#     Returns:
#         audio_out: Convolved audio. Tensor of shape
#             [batch, audio_timesteps].
#     """
#     # Add a frame dimension to impulse response if it doesn't have one.
#     ir_shape = impulse_response.size()
#     if len(ir_shape) == 2:
#         impulse_response = impulse_response.unsqueeze(1)
#         ir_shape = impulse_response.size()

#     # Get shapes of audio and impulse response.
#     batch_size_ir, n_ir_frames, ir_size = ir_shape
#     batch_size, audio_size = audio.size()  # B, T

#     # Validate that batch sizes match.
#     if batch_size != batch_size_ir:
#         raise ValueError(
#             "Batch size of audio ({}) and impulse response ({}) must "
#             "be the same.".format(batch_size, batch_size_ir)
#         )

#     # Cut audio into 50% overlapped frames (center padding).
#     hop_size = int(audio_size / n_ir_frames)
#     frame_size = 2 * hop_size
#     audio_frames = F.pad(audio, (hop_size, hop_size)).unfold(1, frame_size, hop_size)

#     # Apply Bartlett (triangular) window
#     window = torch.bartlett_window(frame_size).to(audio_frames)
#     audio_frames = audio_frames * window

#     # Pad and FFT the audio and impulse responses.
#     fft_size = get_fft_size(frame_size, ir_size, power_of_2=False)
#     audio_fft = torch.fft.rfft(audio_frames, fft_size)
#     ir_fft = torch.fft.rfft(
#         torch.cat((impulse_response, impulse_response[:, -1:, :]), 1), fft_size
#     )

#     # Multiply the FFTs (same as convolution in time).
#     audio_ir_fft = torch.multiply(audio_fft, ir_fft)

#     # Take the IFFT to resynthesize audio.
#     audio_frames_out = torch.fft.irfft(audio_ir_fft, fft_size)

#     # Overlap Add
#     batch_size, n_audio_frames, frame_size = (
#         audio_frames_out.size()
#     )  # # B, n_frames+1, 2*(hop_size+n_mags-1)-1
#     fold = torch.nn.Fold(
#         output_size=(1, (n_audio_frames - 1) * hop_size + frame_size),
#         kernel_size=(1, frame_size),
#         stride=(1, hop_size),
#     )
#     output_signal = fold(audio_frames_out.transpose(1, 2)).squeeze(1).squeeze(1)

#     # Crop and shift the output audio.
#     output_signal = crop_and_compensate_delay(
#         output_signal[:, hop_size:], audio_size, ir_size
#     )
#     return output_signal


# def frequency_filter(audio, magnitudes, hann_window=True, half_width_frames=None):

#     impulse_response = frequency_impulse_response(
#         magnitudes, hann_window, half_width_frames
#     )
#     return fft_convolve(audio, impulse_response)


# def frequency_impulse_response(magnitudes, hann_window=True, half_width_frames=None):

#     # Get the IR
#     impulse_response = torch.fft.irfft(magnitudes)  # B, n_frames, 2*(n_mags-1)

#     # Window and put in causal form.
#     if hann_window:
#         if half_width_frames is None:
#             impulse_response = apply_window_to_impulse_response(impulse_response)
#         else:
#             impulse_response = apply_dynamic_window_to_impulse_response(
#                 impulse_response, half_width_frames
#             )
#     else:
#         impulse_response = impulse_response.roll(impulse_response.size(-1) // 2, -1)
#     return impulse_response


