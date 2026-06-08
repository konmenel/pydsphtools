"""
Practical examples for using JONSWAP spectrum functions
"""
from pydsphtools._waves import (
    jonswap_spectrum_period,
    jonswap_spectrum_frequency,
    free_surface_elevation,
    velocity_2d,
)
import numpy as np
import matplotlib.pyplot as plt

# After adding the functions to your _waves.py, import them:
# from _waves import jonswap_spectrum, free_surface_elevation, velocity_2d


# ============================================================================
# EXAMPLE 1: Generate and plot the JONSWAP spectrum
# ============================================================================


def example_spectrum():
    """Generate and visualize the JONSWAP spectrum."""
    Hs = 2.0  # Significant wave height [m]
    Tp = 8.0  # Peak period [s]

    # Generate spectrum at 200 period points
    periods, spec = jonswap_spectrum_period(Hs=Hs, Tp=Tp, nperiods=200, gamma=3.3)

    # Verify that Hs is respected
    m0 = np.trapezoid(spec, periods)
    Hs_calc = 4 * np.sqrt(m0)

    print(f"Input Hs: {Hs:.2f} m")
    print(f"Calculated Hs: {Hs_calc:.2f} m")
    print(f"Zeroth moment m0: {m0:.4f} m²·s")
    print(f"Peak period: {Tp:.1f} s")

    # Plot spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(periods, spec, "b-", linewidth=2)
    plt.xlabel("Period [s]")
    plt.ylabel("Spectrum [m²·s]")
    plt.title(f"JONSWAP Spectrum (Hs={Hs}m, Tp={Tp}s, γ=3.3)")
    plt.grid(True, alpha=0.3)
    # plt.show()

    Hs = 2.0  # Significant wave height [m]
    Tp = 8.0  # Peak period [s]
    fp = 1.0 / Tp

    # Generate spectrum at 200 period points
    freq, spec = jonswap_spectrum_frequency(Hs=Hs, fp=fp, nfreqs=200, gamma=3.3)

    # Verify that Hs is respected
    m0 = np.trapezoid(spec, freq)
    Hs_calc = 4 * np.sqrt(m0)

    print(f"Input Hs: {Hs:.2f} m")
    print(f"Calculated Hs: {Hs_calc:.2f} m")
    print(f"Zeroth moment m0: {m0:.4f} m²·s")
    print(f"Peak period: {Tp:.1f} s")

    # Plot spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(freq, spec, "b-", linewidth=2)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Spectrum [m²·s]")
    plt.title(f"JONSWAP Spectrum (Hs={Hs}m, fp={fp}Hz, γ=3.3)")
    plt.grid(True, alpha=0.3)
    # plt.show()


# ============================================================================
# EXAMPLE 2: Time series of surface elevation at a fixed point
# ============================================================================


def example_elevation_timeseries():
    """Generate surface elevation time series at a fixed location."""
    Hs = 2.0
    Tp = 8.0
    depth = 20.0
    x_fixed = 10.0  # Fixed horizontal position [m]

    # Generate spectrum once
    periods, spec = jonswap_spectrum_period(Hs=Hs, Tp=Tp, nperiods=100, gamma=3.3)

    # Generate 60 seconds of data at 10 Hz sampling
    t = np.arange(0, 60, 0.1)  # Time array [s]

    # Calculate surface elevation from the spectrum
    eta = free_surface_elevation(
        periods, spec, x=x_fixed, t=t, depth=depth, second_order=False, random_seed=42
    )

    # Plot time series
    plt.figure(figsize=(12, 5))
    plt.plot(t, eta[0, :], "b-", linewidth=1)
    plt.xlabel("Time [s]")
    plt.ylabel("Surface Elevation [m]")
    plt.title(f"Surface Elevation at x={x_fixed}m (Hs={Hs}m, Tp={Tp}s)")
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color="k", linestyle="--", alpha=0.3)
    # plt.show()

    # Calculate wave statistics
    print("Time series statistics:")
    print(f"  Mean: {np.mean(eta):.3f} m")
    print(f"  Std Dev: {np.std(eta):.3f} m")
    print(f"  Max: {np.max(eta):.3f} m")
    print(f"  Min: {np.min(eta):.3f} m")
    print(f"  Hs (4·σ): {4 * np.std(eta):.3f} m")


# ============================================================================
# EXAMPLE 3: Spatial-temporal surface elevation field (snapshot)
# ============================================================================


def example_elevation_field():
    """Generate a spatial-temporal snapshot of surface elevation."""
    Hs = 1.5
    Tp = 7.0
    depth = 30.0

    # Generate spectrum once
    periods, spec = jonswap_spectrum_period(Hs=Hs, Tp=Tp, nperiods=150, gamma=3.3)

    # Spatial and temporal grid
    x = np.linspace(0, 150, 100)  # 150m domain
    t_snapshot = np.array([5.0, 6.0, 7.0, 8.0])  # Four time snapshots [s]

    eta = free_surface_elevation(
        periods,
        spec,
        x=x,
        t=t_snapshot,
        depth=depth,
        second_order=False,
        random_seed=123,
    )

    # Plot four snapshots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, t_val in enumerate(t_snapshot):
        axes[i].plot(x, eta[:, i], "b-", linewidth=2)
        axes[i].set_xlabel("Position x [m]")
        axes[i].set_ylabel("Elevation [m]")
        axes[i].set_title(f"Surface at t={t_val}s")
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(0, color="k", linestyle="--", alpha=0.3)
        axes[i].set_ylim([-3, 3])

    plt.tight_layout()
    # plt.show()


# ============================================================================
# EXAMPLE 4: Vertical velocity profile at a fixed location
# ============================================================================


def example_velocity_profile():
    """Calculate velocity at different depths (vertical profile)."""
    Hs = 2.0
    Tp = 8.0
    depth = 20.0
    x_fixed = 25.0  # Fixed horizontal position [m]
    t_fixed = 10.0  # Fixed time [s]

    # Generate spectrum once
    periods, spec = jonswap_spectrum_period(Hs=Hs, Tp=Tp, nperiods=100, gamma=3.3)

    # Vertical positions relative to still water surface
    # z=0 at surface, z=-depth at bottom
    z = np.linspace(0, -depth, 50)

    # Calculate velocity components
    u, w = velocity_2d(
        periods,
        spec,
        x=x_fixed,
        z=z,
        t=t_fixed,
        depth=depth,
        second_order=False,
        random_seed=42,
    )

    # Plot vertical profiles
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Horizontal velocity profile
    ax1.plot(u[0, :, 0], z, "b-", linewidth=2)
    ax1.set_xlabel("Horizontal Velocity u [m/s]")
    ax1.set_ylabel("Depth z [m]")
    ax1.set_title(f"Horizontal Velocity Profile at t={t_fixed}s")
    ax1.grid(True, alpha=0.3)
    ax1.axvline(0, color="k", linestyle="--", alpha=0.3)

    # Vertical velocity profile
    ax2.plot(w[0, :, 0], z, "r-", linewidth=2)
    ax2.set_xlabel("Vertical Velocity w [m/s]")
    ax2.set_ylabel("Depth z [m]")
    ax2.set_title(f"Vertical Velocity Profile at t={t_fixed}s")
    ax2.grid(True, alpha=0.3)
    ax2.axvline(0, color="k", linestyle="--", alpha=0.3)

    plt.tight_layout()
    # plt.show()


# ============================================================================
# EXAMPLE 5: Time series of velocity at different depths
# ============================================================================


def example_velocity_timeseries():
    """Generate velocity time series at different depths."""
    Hs = 2.0
    Tp = 8.0
    depth = 20.0
    x_fixed = 10.0

    # Generate spectrum once
    periods, spec = jonswap_spectrum_period(Hs=Hs, Tp=Tp, nperiods=100, gamma=3.3)

    t = np.arange(0, 40, 0.1)  # 40 seconds at 10 Hz

    # Evaluate at three different depths
    z_depths = np.array([0.0, -5.0, -15.0])  # Surface, mid-depth, near-bottom

    # Calculate velocities
    u, w = velocity_2d(
        periods,
        spec,
        x=x_fixed,
        z=z_depths,
        t=t,
        depth=depth,
        second_order=False,
        random_seed=42,
    )

    # Plot horizontal and vertical velocity at different depths
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    labels = [f"z={z:.1f}m" for z in z_depths]
    colors = ["blue", "green", "red"]

    for i, (label, color) in enumerate(zip(labels, colors)):
        ax1.plot(t, u[0, i, :], label=label, color=color, linewidth=1, alpha=0.8)
        ax2.plot(t, w[0, i, :], label=label, color=color, linewidth=1, alpha=0.8)

    ax1.set_ylabel("Horizontal Velocity u [m/s]")
    ax1.set_title(f"Horizontal Velocity Time Series (Hs={Hs}m, Tp={Tp}s)")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax1.legend()

    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Vertical Velocity w [m/s]")
    ax2.set_title("Vertical Velocity Time Series")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    # plt.show()


# ============================================================================
# EXAMPLE 6: Second-order vs First-order comparison
# ============================================================================


def example_second_order_comparison():
    """Compare first-order and second-order wave theory."""
    Hs = 3.0  # Steeper waves to show second-order effects
    Tp = 6.0
    depth = 15.0
    x_fixed = 20.0

    # Generate spectrum once
    periods, spec = jonswap_spectrum_period(Hs=Hs, Tp=Tp, nperiods=150, gamma=3.3)

    t = np.arange(0, 30, 0.1)

    # First-order calculation
    eta_1st = free_surface_elevation(
        periods, spec, x=x_fixed, t=t, depth=depth, second_order=False, random_seed=42
    )

    # Second-order calculation
    eta_2nd = free_surface_elevation(
        periods, spec, x=x_fixed, t=t, depth=depth, second_order=True, random_seed=42
    )

    # Plot comparison
    plt.figure(figsize=(14, 6))
    plt.plot(t, eta_1st[0, :], "b-", label="First-order (Airy)", linewidth=1.5, alpha=0.8)
    plt.plot(t, eta_2nd[0, :], "r--", label="Second-order (Stokes)", linewidth=1.5, alpha=0.8)
    plt.xlabel("Time [s]")
    plt.ylabel("Surface Elevation [m]")
    plt.title(
        f"First-order vs Second-order Wave Theory (Hs={Hs}m, Tp={Tp}s, h={depth}m)"
    )
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color="k", linestyle="--", alpha=0.3)
    # plt.show()

    # Calculate difference
    difference = eta_2nd - eta_1st
    print("Second-order Effects:")
    print(f"  Max difference: {np.max(np.abs(difference)):.4f} m")
    print(f"  Mean absolute difference: {np.mean(np.abs(difference)):.4f} m")
    print(f"  RMS difference: {np.sqrt(np.mean(difference**2)):.4f} m")


# ============================================================================
# EXAMPLE 7: Energy spectrum from JONSWAP
# ============================================================================


def example_wave_energy():
    """Analyze wave energy distribution."""
    Hs = 2.0
    Tp = 8.0

    # Generate high-resolution spectrum
    periods, spec = jonswap_spectrum_period(Hs=Hs, Tp=Tp, nperiods=500, gamma=3.3)

    # Convert to spectral moment
    m0 = np.trapezoid(spec, periods)

    # Calculate cumulative energy
    cum_spec = np.cumsum(spec) / np.sum(spec) * 100

    # Find period ranges containing 50%, 90% energy
    idx_50 = np.argmin(np.abs(cum_spec - 50))
    idx_90 = np.argmin(np.abs(cum_spec - 90))
    T_50 = periods[idx_50]
    T_90 = periods[idx_90]

    # Plot spectrum with energy bands
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Spectrum
    ax1.plot(periods, spec, "b-", linewidth=2)
    ax1.axvline(Tp, color="r", linestyle="--", label=f"Peak period = {Tp:.2f} s")
    ax1.set_xlabel("Period [s]")
    ax1.set_ylabel("Spectrum [m²·s]")
    ax1.set_title(f"JONSWAP Spectrum (Hs={Hs}m, Tp={Tp}s)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Cumulative energy
    ax2.plot(periods, cum_spec, "g-", linewidth=2)
    ax2.axhline(50, color="r", linestyle="--", alpha=0.5, label="50%")
    ax2.axhline(90, color="r", linestyle="--", alpha=0.5, label="90%")
    ax2.axvline(T_50, color="r", linestyle=":", alpha=0.5)
    ax2.axvline(T_90, color="r", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Period [s]")
    ax2.set_ylabel("Cumulative Energy [%]")
    ax2.set_title("Cumulative Energy Distribution")
    ax2.set_ylim([0, 100])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.show()

    print("Energy Distribution:")
    print(f"  50% of energy: 0 - {T_50:.3f} s")
    print(f"  90% of energy: 0 - {T_90:.3f} s")
    print(f"  Period bandwidth: {T_90 - T_50:.3f} s")


# ============================================================================
# Run examples
# ============================================================================

if __name__ == "__main__":
    print("JONSWAP Spectrum Examples")
    print("=" * 50)

    example_spectrum()
    print("=" * 50)
    example_elevation_timeseries()
    print("=" * 50)
    example_elevation_field()
    example_velocity_profile()
    example_velocity_timeseries()
    example_second_order_comparison()
    print("=" * 50)
    example_wave_energy()
    print("=" * 50)

    plt.show()

    print("\nExamples are ready to use. Uncomment one in the if __name__ block.")
