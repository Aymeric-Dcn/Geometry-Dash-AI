"""
Replay a solution and turn it into a GIF (no game or pygame needed).
Usage: python render.py solution_qlearning_stereo_lite_practice_state.txt [level_name]
"""
import sys

from PIL import Image, ImageDraw

from gdsim import GDEnv, PHYS_HZ, TICKS_PER_STEP

PX = 28                 # pixels per block
W, H = 22 * PX, 7 * PX  # 22 x 7 block window
GROUND = H - PX
BG, GROUND_C, BLOCK_C, SPIKE_C = (24, 32, 72), (40, 70, 160), (30, 30, 40), (235, 235, 245)
PLAYER_C, TRAIL_C = (120, 230, 90), (255, 220, 80)


def to_px(x, y, cam):
    return (x - cam) * PX, GROUND - y * PX


def frame(env, traj_upto, cam, title):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, GROUND, W, H], fill=GROUND_C)
    lv = env.level
    for col in range(int(cam) - 1, int(cam) + 24):
        for bx, by in lv.blocks.get(col, ()):
            x0, y0 = to_px(bx, by + 1, cam)
            d.rectangle([x0, y0, x0 + PX - 1, y0 + PX - 1], fill=BLOCK_C, outline=(200, 200, 230))
        for sx, sy in lv.spikes.get(col, ()):
            x0, y0 = to_px(sx, sy, cam)
            d.polygon([(x0 + 2, y0), (x0 + PX - 2, y0), (x0 + PX / 2, y0 - PX + 4)], fill=SPIKE_C)
    pts = [to_px(s.x + 0.5, s.y + 0.5, cam) for s in traj_upto[-60:]]
    if len(pts) > 1:
        d.line(pts, fill=TRAIL_C, width=2)
    s = traj_upto[-1]
    x0, y0 = to_px(s.x, s.y + 1, cam)
    d.rectangle([x0, y0, x0 + PX - 1, y0 + PX - 1], fill=(220, 60, 60) if s.dead else PLAYER_C,
                outline=(0, 0, 0), width=2)
    pct = min(100, int(100 * s.x / lv.end_x))
    d.text((8, 6), f"{title}   {pct}%", fill=(255, 255, 255))
    return img


def render(actions, level_name, out_path, title="", every=2):
    env = GDEnv(level_name)
    env.reset()
    traj = [env.state]
    frames = []
    for i, a in enumerate(actions):
        env.step(a)
        traj.append(env.state)
        if i % every == 0 or env.state.dead or env.state.won:
            cam = max(0.0, env.state.x - 6)
            frames.append(frame(env, traj, cam, title))
        if env.state.dead or env.state.won:
            break
    frames += [frames[-1]] * 20       # short pause at the end
    ms = int(1000 * every * TICKS_PER_STEP / PHYS_HZ)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=ms, loop=0)
    return env.state


if __name__ == "__main__":
    path = sys.argv[1]
    level = sys.argv[2] if len(sys.argv) > 2 else "stereo_lite"
    acts = [int(c) for c in open(path).read().strip()]
    out = path.rsplit(".", 1)[0] + ".gif"
    s = render(acts, level, out, title=level)
    print(f"GIF written: {out}  ({'won' if s.won else 'died' if s.dead else 'incomplete'})")
