
from ceviche_challenges import waveguide_bend, mode_converter
from ceviche_challenges import units as u

import invrs_gym.challenges.ceviche.challenge as inv_ch
import ceviche_challenges as cc 
from invrs_gym.challenges.ceviche import defaults

CUSTOM_WAVEGUIDE_BEND_SPEC = waveguide_bend.spec.WaveguideBendSpec(
    wg_width=defaults.WG_WIDTH,
    wg_length=defaults.WG_LENGTH,
    wg_mode_padding=defaults.WG_MODE_PADDING,
    padding=defaults.PADDING,
    port_pml_offset=defaults.PORT_PML_OFFSET,
    variable_region_size=(2000 * u.nm, 2000 * u.nm),
    cladding_permittivity=defaults.CLADDING_PERMITTIVITY,
    slab_permittivity=defaults.SLAB_PERMITTIVITY,
    input_monitor_offset=defaults.INPUT_MONITOR_OFFSET,
    pml_width=defaults.PML_WIDTH_GRIDPOINTS,
)

CUSTOM_MODE_CONVERTER_SPEC = mode_converter.spec.ModeConverterSpec(
    left_wg_width=defaults.WG_WIDTH,
    left_wg_mode_padding=defaults.WG_MODE_PADDING,
    left_wg_mode_order=1,  # Fundamental mode.
    right_wg_width=defaults.WG_WIDTH,
    right_wg_mode_padding=defaults.WG_MODE_PADDING,
    right_wg_mode_order=2,  # Second mode.
    wg_length=defaults.WG_LENGTH,
    padding=defaults.PADDING,
    port_pml_offset=defaults.PORT_PML_OFFSET,
    variable_region_size=(2000 * u.nm, 2000 * u.nm),
    cladding_permittivity=defaults.CLADDING_PERMITTIVITY,
    slab_permittivity=defaults.SLAB_PERMITTIVITY,
    input_monitor_offset=defaults.INPUT_MONITOR_OFFSET,
    pml_width=defaults.PML_WIDTH_GRIDPOINTS,
)


def custom_lightweight_waveguide_bend_challenge(mesh_resolution_nm):
  return inv_ch.CevicheChallenge(
        component= inv_ch.CevicheComponent(
            ceviche_model=cc.waveguide_bend.model.WaveguideBendModel(
                params=cc.params.CevicheSimParams(
                    resolution=mesh_resolution_nm * u.nm,
                    wavelengths=u.Array(defaults.LIGHTWEIGHT_WAVELENGTHS_NM, u.nm),
                ),
                spec=CUSTOM_WAVEGUIDE_BEND_SPEC,
            ),
            symmetries=defaults.WAVEGUIDE_BEND_SYMMETRIES,
            minimum_width=defaults.LIGHTWEIGHT_MINIMUM_WIDTH,
            minimum_spacing=defaults.LIGHTWEIGHT_MINIMUM_SPACING,
            density_initializer=inv_ch.density_initializer,
        ),
        transmission_lower_bound=defaults.WAVEGUIDE_BEND_TRANSMISSION_LOWER_BOUND,
        transmission_upper_bound=defaults.WAVEGUIDE_BEND_TRANSMISSION_UPPER_BOUND,
  )

def custom_lightweight_modeconvertor_challenge(mesh_resolution_nm):
  return inv_ch.CevicheChallenge(
        component=inv_ch.CevicheComponent(
            ceviche_model=cc.mode_converter.model.ModeConverterModel(
                params=cc.params.CevicheSimParams(
                    resolution=mesh_resolution_nm * u.nm,
                    wavelengths=u.Array(defaults.LIGHTWEIGHT_WAVELENGTHS_NM, u.nm),
                ),
                spec= CUSTOM_MODE_CONVERTER_SPEC,
            ),
            minimum_width=defaults.LIGHTWEIGHT_MINIMUM_WIDTH,
            minimum_spacing=defaults.LIGHTWEIGHT_MINIMUM_SPACING,
            density_initializer=inv_ch.density_initializer,
        ),
        transmission_lower_bound=defaults.MODE_CONVERTER_TRANSMISSION_LOWER_BOUND,
        transmission_upper_bound=defaults.MODE_CONVERTER_TRANSMISSION_UPPER_BOUND,
  )

