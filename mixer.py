import numpy as np
import sounddevice as sd
import threading
import argparse
import random
import time
import math
import curses
import json  # new import
import os  # new import

PRESET_FILE = "~/.preset.json"  # new preset path

# Global state for effective rate (for brown noise)
base_rate = None  # set in main()
current_eff_rate = None  # set in main()
target_eff_rate = None  # set in main()

# Global states for brown noise (alternative method)
last_brown = 0.0

# Global states for pink noise (Paul Kellet’s approximation)
pink_b0 = pink_b1 = pink_b2 = pink_b3 = pink_b4 = pink_b5 = pink_b6 = 0.0

# Add new global states for 'blue' and 'violet' noise generation
blue_prev = 0.0
violet_prev = 0.0

# Global mix settings for each noise type
mix_settings = {
    "white": {"mix": 0.2, "osc": False, "osc_phase": 0.0},
    "pink": {"mix": 0.2, "osc": False, "osc_phase": 0.0},
    "brown": {"mix": 0.6, "osc": False, "osc_phase": 0.0},
    "blue": {"mix": 0.0, "osc": False, "osc_phase": 0.0},
    "violet": {"mix": 0.0, "osc": False, "osc_phase": 0.0},
    "grey": {"mix": 0.2, "osc": False, "osc_phase": 0.0},
    "green": {"mix": 0.0, "osc": False, "osc_phase": 0.0},
    "black": {"mix": 0.0, "osc": False, "osc_phase": 0.0},
    "brown.2": {"mix": 0.0, "osc": False, "osc_phase": 0.0},
    "brown.3": {"mix": 0.0, "osc": False, "osc_phase": 0.0},
    "brown.4": {"mix": 0.0, "osc": False, "osc_phase": 0.0},
}
osc_freq = 0.1  # slower oscillation frequency
log_message = ""  # global log message to display in UI
types_order = [
    "white",
    "pink",
    "brown",
    "brown.2",
    "brown.3",
    "brown.4",
    "blue",
    "violet",
    "grey",
    "green",
    "black",
]
selected_index = 0  # for slider UI
paused = False  # new global pause flag
last_waveform = None  # new global variable for oscilloscope data


def multi_noise_callback(outdata, frames, time_info, status):
    global current_eff_rate, target_eff_rate, base_rate, last_brown, paused, last_waveform
    global pink_b0, pink_b1, pink_b2, pink_b3, pink_b4, pink_b5, pink_b6
    global blue_prev, violet_prev

    if status:
        print(status)
    # Check pause flag
    if paused:
        outdata.fill(0)
        return
    smoothing = 0.0
    current_eff_rate += smoothing * (target_eff_rate - current_eff_rate)
    scale = current_eff_rate / base_rate

    # Generate base white noise
    white = np.random.randn(frames)

    # Pink noise remains unchanged (Paul Kellet’s approximation)
    pink = np.empty(frames)
    for i in range(frames):
        white_sample = np.random.randn()
        pink_b0 = 0.99886 * pink_b0 + white_sample * 0.0555179
        pink_b1 = 0.99332 * pink_b1 + white_sample * 0.0750759
        pink_b2 = 0.96900 * pink_b2 + white_sample * 0.1538520
        pink_b3 = 0.86650 * pink_b3 + white_sample * 0.3104856
        pink_b4 = 0.55000 * pink_b4 + white_sample * 0.5329522
        pink_b5 = -0.7616 * pink_b5 - white_sample * 0.0168980
        curr_pink = (
            pink_b0
            + pink_b1
            + pink_b2
            + pink_b3
            + pink_b4
            + pink_b5
            + pink_b6
            + white_sample * 0.5362
        )
        pink[i] = curr_pink
        pink_b6 = white_sample * 0.115926

    # Brown noise: revert to the previously used algorithm with increased alpha
    brown = np.empty(frames)
    alpha = 0.9  # increased from 0.02 for stronger integration
    for i in range(frames):
        step = np.random.randn() * 0.1 * scale
        last_brown = alpha * step + (1 - alpha) * last_brown
        brown[i] = last_brown

    # Additional Brown algorithms:
    # brown.2: cumulative sum of white noise (traditional Brown)
    brown2 = np.cumsum(white)
    brown2 = brown2 - np.mean(brown2)
    if np.max(np.abs(brown2)) != 0:
        brown2 = brown2 / np.max(np.abs(brown2))

    # brown.3: leaky integrator with periodic modulation
    brown3 = np.empty(frames)
    state = 0.0
    for i in range(frames):
        step = np.random.randn() * 0.1 * scale
        state = 0.95 * state + step
        # Add low-frequency oscillation (periodic modulation)
        brown3[i] = state + 0.1 * math.sin(2 * math.pi * i / frames)

    # brown.4: standard brown modulated by a low-frequency sinusoid
    brown4 = brown * (0.5 + 0.5 * np.sin(2 * math.pi * np.arange(frames) / frames))

    # Blue noise: first difference of white noise
    blue = np.concatenate(([0], np.diff(white)))

    # Violet noise: first difference of blue noise
    violet = np.concatenate(([0], np.diff(blue)))

    # Grey noise: scaled white noise
    grey = 0.8 * white

    # Green noise: moving average (low-pass filter) of white noise
    kernel = np.ones(10) / 10.0
    green = np.convolve(white, kernel, mode="same")

    # Black noise: near-silence: greatly attenuated white noise
    black = 0.01 * white

    fixed_gains = {
        "white": 0.3,
        "pink": 0.3,
        "brown": 1.0,  # increased brown gain from 0.3 to 1.0
        "brown.2": 1.0,
        "brown.3": 1.0,
        "brown.4": 1.0,
        "blue": 0.3,
        "violet": 0.3,
        "grey": 0.3,
        "green": 0.3,
        "black": 0.3,
    }

    dt = frames / base_rate
    final = np.zeros(frames)
    for ntype, noise_array in zip(
        [
            "white",
            "pink",
            "brown",
            "brown.2",
            "brown.3",
            "brown.4",
            "blue",
            "violet",
            "grey",
            "green",
            "black",
        ],
        [white, pink, brown, brown2, brown3, brown4, blue, violet, grey, green, black],
    ):
        setting = mix_settings[ntype]
        mix_val = setting["mix"]
        if setting["osc"]:
            mod = 0.5 + 0.5 * math.sin(setting["osc_phase"])
            mix_val *= mod
            setting["osc_phase"] += 2 * math.pi * osc_freq * dt
        final += fixed_gains.get(ntype, 0.3) * mix_val * noise_array

    outdata[:] = final.reshape(-1, 1)
    # Store a copy for the oscilloscope
    last_waveform = final.copy()


def create_callback():
    return multi_noise_callback


def save_preset():
    # Save mix_settings to PRESET_FILE
    global mix_settings
    try:
        with open(PRESET_FILE, "w") as f:
            json.dump(mix_settings, f)
    except Exception as e:
        print("Error saving preset:", e)


def load_preset():
    # Load mix_settings from PRESET_FILE if exists
    global mix_settings
    if os.path.exists(PRESET_FILE):
        try:
            with open(PRESET_FILE, "r") as f:
                preset = json.load(f)
            mix_settings.update(preset)
        except Exception as e:
            print("Error loading preset:", e)


def randomize_samplerate_thread(stop_event):
    global target_eff_rate, log_message
    while not stop_event.is_set():
        time.sleep(20)
        new_rate = random.randint(1200, 2000)
        target_eff_rate = new_rate
        log_message = (
            f"Effective rate changed to {new_rate}"  # update log instead of printing
        )


def slider_ui(stdscr, stop_event):
    global selected_index, log_message, paused, last_waveform
    curses.curs_set(0)
    curses.start_color()  # initialize colors
    # Fixed color bands for rows:
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)  # bottom third: green
    curses.init_pair(
        2, curses.COLOR_YELLOW, curses.COLOR_BLACK
    )  # middle third: orange (magenta substitute)
    curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)  # top third: red
    stdscr.nodelay(True)
    stdscr.keypad(True)
    max_width = 50
    refresh_rate = 0.1
    label_width = max(len(x) for x in types_order)
    usage_text = (
        "Up/Down:select, Left/Right:mix, 'o':toggle osc, 's':save preset, "
        "'z':zero, SPACE:pause/unpause, 'q':quit, 'r':randomize"
    )
    while not stop_event.is_set():
        stdscr.erase()
        stdscr.border()
        stdscr.addstr(2, 2, usage_text)
        for idx, ntype in enumerate(types_order):
            setting = mix_settings[ntype]
            mix_val = setting["mix"]
            bar_len = int(mix_val * max_width)
            bar = "#" * bar_len + "-" * (max_width - bar_len)
            line = f"{ntype.ljust(label_width)}: [{bar}] {mix_val:4.2f}  Osc: {'ON' if setting['osc'] else 'OFF'}"
            stdscr.addstr(
                4 + idx, 2, line, curses.A_REVERSE if idx == selected_index else 0
            )

        # Draw oscilloscope panel under mixer.
        osc_start_row = 4 + len(types_order) + 1
        osc_height = 10
        osc_width = max_width  # match mix panel width
        osc_win = stdscr.derwin(osc_height, osc_width, osc_start_row, 2)
        osc_win.erase()
        if last_waveform is not None:
            # Smooth and downsample together:
            window = np.ones(5) / 5.0
            smoothed = np.convolve(last_waveform, window, mode="same")
            x_indices = np.linspace(0, len(smoothed) - 1, osc_width)
            sampled = np.interp(x_indices, np.arange(len(smoothed)), smoothed)
            window_size = 3  # use columns x-1 to x+1 for range calculation
            for x in range(osc_width):
                # Current sample point.
                sample = sampled[x]
                point_y = int((sample + 1) / 2 * osc_height)
                # Compute range over a small window.
                w_start = max(0, x - 1)
                w_end = min(osc_width, x + 2)
                window_vals = sampled[w_start:w_end]
                rmin = min(window_vals)
                rmax = max(window_vals)
                y_min = int((rmin + 1) / 2 * osc_height)
                y_max = int((rmax + 1) / 2 * osc_height)
                # Draw vertical range bar (using block '█') for rows between y_min and y_max.
                for y in range(y_min, y_max + 1):
                    # Fixed color assignment based on row index.
                    if y < osc_height / 3:
                        bar_color = curses.color_pair(3)  # red top
                    elif y < 2 * osc_height / 3:
                        bar_color = curses.color_pair(2)  # orange middle
                    else:
                        bar_color = curses.color_pair(1)  # green bottom
                    try:
                        osc_win.addch(y, x, "•", bar_color)
                    except curses.error:
                        pass
                # Draw a single point for the current sample.
                if point_y >= osc_height:
                    point_y = osc_height - 1
                if point_y < 0:
                    point_y = 0
                # Use the fixed color based on point_y.
                if point_y < osc_height / 3:
                    pt_color = curses.color_pair(3)
                elif point_y < 2 * osc_height / 3:
                    pt_color = curses.color_pair(2)
                else:
                    pt_color = curses.color_pair(1)
                try:
                    osc_win.addch(point_y, x, "•", pt_color)
                except curses.error:
                    pass
        osc_win.border()
        osc_win.noutrefresh()

        # Add vertical amplitude labels beside the oscilloscope.
        stdscr.addstr(osc_start_row + 1, 0, "1.0")
        stdscr.addstr(osc_start_row + osc_height // 2, 0, "0.5")
        stdscr.addstr(osc_start_row + osc_height - 1, 0, "0.0")

        stdscr.addstr(curses.LINES - 2, 2, log_message.ljust(curses.COLS - 4))
        stdscr.noutrefresh()
        curses.doupdate()

        try:
            key = stdscr.getch()
        except:
            key = -1
        if key != -1:
            if key == curses.KEY_UP:
                selected_index = (selected_index - 1) % len(types_order)
            elif key == curses.KEY_DOWN:
                selected_index = (selected_index + 1) % len(types_order)
            elif key == curses.KEY_LEFT:
                curr = mix_settings[types_order[selected_index]]["mix"]
                mix_settings[types_order[selected_index]]["mix"] = max(0.0, curr - 0.05)
            elif key == curses.KEY_RIGHT:
                curr = mix_settings[types_order[selected_index]]["mix"]
                mix_settings[types_order[selected_index]]["mix"] = min(1.0, curr + 0.05)
            elif key == ord("o"):
                curr = mix_settings[types_order[selected_index]]["osc"]
                mix_settings[types_order[selected_index]]["osc"] = not curr
            elif key == ord("s"):
                save_preset()
                log_message = "Preset saved."
            elif key == ord("z"):
                for k in mix_settings:
                    mix_settings[k]["mix"] = 0.0
                log_message = "All mixer bars zeroed."
            elif key == ord(" "):
                paused = not paused
                log_message = "Paused." if paused else "Unpaused."
            elif key == ord("r"):
                # Randomize noise type (mix_settings) values.
                keys_list = list(mix_settings.keys())
                # Choose one key to have a dominant mix (between 0.5 and 1.0).
                dominant_key = random.choice(keys_list)
                for ntype in mix_settings:
                    if ntype == dominant_key:
                        mix_settings[ntype]["mix"] = random.uniform(0.5, 1.0)
                    else:
                        mix_settings[ntype]["mix"] = random.uniform(0.0, 0.5)
                # Set oscillator for at most two noise types.
                oscillate_keys = random.sample(keys_list, min(2, len(keys_list)))
                for ntype in mix_settings:
                    mix_settings[ntype]["osc"] = ntype in oscillate_keys
                    mix_settings[ntype]["osc_phase"] = random.random() * 2 * math.pi
                # Randomize global oscillator frequency.
                global osc_freq
                osc_freq = random.uniform(0.05, 0.2)
                log_message = "Randomized noise type settings."
            elif key == ord("q"):
                stop_event.set()
        time.sleep(refresh_rate)


def slider_ui_thread(stop_event):
    curses.wrapper(slider_ui, stop_event)


def main():
    global base_rate, current_eff_rate, target_eff_rate
    parser = argparse.ArgumentParser(
        description="Multi-noise Player with adjustable mix/oscillation (white, pink, brown)."
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=1200,
        help="Initial effective noise rate (Hz).",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        default=True,
        help="Randomly change effective noise rate every 20 seconds (default on).",
    )
    args = parser.parse_args()

    base_rate = args.samplerate
    current_eff_rate = base_rate
    target_eff_rate = base_rate

    load_preset()  # load preset on start

    stop_event = threading.Event()
    if args.randomize:
        sr_thread = threading.Thread(
            target=randomize_samplerate_thread, args=(stop_event,), daemon=True
        )
        sr_thread.start()
    # Removed key_thread that previously waited for Enter
    ui_thread = threading.Thread(
        target=slider_ui_thread, args=(stop_event,), daemon=True
    )
    ui_thread.start()

    stream = sd.OutputStream(
        channels=1,
        callback=create_callback(),
        samplerate=base_rate,
        blocksize=1024,
    )
    stream.start()
    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    main()
