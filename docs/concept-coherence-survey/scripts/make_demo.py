"""Generate clearly synthetic LOCAL TEST assets; never replaces existing folders."""
import argparse
import random
from pathlib import Path
from PIL import Image, ImageDraw

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('destination',type=Path)
args=parser.parse_args()
if args.destination.exists():
    raise SystemExit('Destination already exists. Choose a new empty path.')
rng=random.Random(42)
for g in range(1,4):
    for p in range(1,7):
        patch=Image.new('RGB',(128,128),(rng.randrange(80,130),rng.randrange(95,150),rng.randrange(90,150)))
        d=ImageDraw.Draw(patch)
        if g==1:
            for y in range(-10,140,19): d.line((0,y,128,y+8),fill=(220,225,204),width=6)
        elif g==2:
            d.rectangle((0,0,128,128),fill=(202+rng.randrange(35),115+rng.randrange(25),41))
            d.rectangle((0,0,23+rng.randrange(12),128),fill=(35,36,32))
        else:
            d.polygon([(0,0),(128,0),(128,128)],fill=(224,222,206))
            for i in range(200):
                x,y=rng.randrange(128),rng.randrange(128)
                if x<y: d.point((x,y),fill=(155,155,152))
        frame=Image.new('RGB',(640,400),(190,200,184))
        fd=ImageDraw.Draw(frame)
        fd.text((20,20),'SYNTHETIC DEMO FRAME - NOT RESEARCH DATA',fill=(30,50,40))
        x,y=rng.randrange(30,450),rng.randrange(60,230)
        frame.paste(patch,(x,y));fd.rectangle((x-2,y-2,x+130,y+130),outline='red',width=3)
        folder=args.destination/f'group_{g}'/f'patch_{p}'
        folder.mkdir(parents=True)
        patch.save(folder/'patch.png');frame.save(folder/'target_frame_boxed.jpg')
print(f'Created 3 synthetic groups at {args.destination.resolve()}')
