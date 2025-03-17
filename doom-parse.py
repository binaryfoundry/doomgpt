import struct
import os
import json
import omg  # pip install omgifol
from dataclasses import dataclass

# Input & output directories
input_dir = "./wads"

@dataclass
class Vertex:
    x: int
    y: int

@dataclass
class Linedef:
    v0: int
    v1: int
    flags: int
    special: int
    tag: int
    side0: int
    side1: int

@dataclass
class Sidedef:
    x_offset: int
    y_offset: int
    upper: str
    lower: str
    middle: str
    sector: int

@dataclass
class Sector:
    floor_height: int
    ceiling_height: int
    floor_texture: str
    ceiling_texture: str
    light_level: int
    type: int
    tag: int

@dataclass
class Thing:
    x_pos: int
    y_pos: int
    angle: int
    type: int
    flags: int

@dataclass
class Map:
    name: str
    vertices: list[Vertex]
    linedefs: list[Linedef]
    sidedefs: list[Sidedef]
    sectors: list[Sector]
    things: list[Thing]
    origin: Vertex

def read_vertices(name_group):
    if "VERTEXES" not in name_group:
        raise KeyError("VERTEXES lump not found in the NameGroup.")

    vertexes_lump = name_group["VERTEXES"].data
    num_vertices = len(vertexes_lump) // 4
    vertices = []

    for i in range(num_vertices):
        offset = i * 4
        x, y = struct.unpack_from("<hh", vertexes_lump, offset)
        vertices.append(Vertex(x, y))

    return vertices

def read_linedefs(name_group):
    if "LINEDEFS" not in name_group:
        raise KeyError("LINEDEFS lump not found in the NameGroup.")

    linedefs_lump = name_group["LINEDEFS"].data
    num_linedefs = len(linedefs_lump) // 14
    linedefs = []

    for i in range(num_linedefs):
        offset = i * 14
        v0, v1, flags, special, tag, side0, side1 = struct.unpack_from("<hhhhhhh", linedefs_lump, offset)
        side0 = side0 if side0 != 0xFFFF else -1
        side1 = side1 if side1 != 0xFFFF else -1

        linedefs.append(Linedef(v0, v1, flags, special, tag, side0, side1))

    return linedefs

def read_sidedefs(name_group):
    if "SIDEDEFS" not in name_group:
        raise KeyError("SIDEDEFS lump not found in the NameGroup.")

    sidedefs_lump = name_group["SIDEDEFS"].data
    num_sidedefs = len(sidedefs_lump) // 30
    sidedefs = []

    for i in range(num_sidedefs):
        offset = i * 30
        x_offset, y_offset, upper, lower, middle, sector = struct.unpack_from("<hh8s8s8sh", sidedefs_lump, offset)

        sidedefs.append(Sidedef(
            x_offset,
            y_offset,
            upper.decode("latin-1").rstrip("\x00"),
            lower.decode("latin-1").rstrip("\x00"),
            middle.decode("latin-1").rstrip("\x00"),
            sector
        ))

    return sidedefs

def read_sectors(name_group):
    if "SECTORS" not in name_group:
        raise KeyError("SECTORS lump not found.")

    sectors_lump = name_group["SECTORS"].data
    num_sectors = len(sectors_lump) // 26
    sectors = []

    for i in range(num_sectors):
        offset = i * 26
        floor_height, ceiling_height, floor_texture, ceiling_texture, light_level, type, tag = struct.unpack_from("<hh8s8shHH", sectors_lump, offset)

        sectors.append(Sector(
            floor_height,
            ceiling_height,
            floor_texture.decode("latin-1").rstrip("\x00"),
            ceiling_texture.decode("latin-1").rstrip("\x00"),
            light_level,
            type,
            tag
        ))

    return sectors

def read_things(name_group):
    if "THINGS" not in name_group:
        raise KeyError("THINGS lump not found.")

    things_lump = name_group["THINGS"].data
    num_things = len(things_lump) // 10
    things = []

    for i in range(num_things):
        offset = i * 10
        x_pos, y_pos, angle, type, flags = struct.unpack_from("<hhHhh", things_lump, offset)
        things.append(Thing(x_pos, y_pos, angle, type, flags))

    return things

def read_origin(things):
    origin = Vertex(0, 0)

    for thing in things:
        if thing.type == 1:
            origin = Vertex(thing.x_pos, thing.y_pos)

    return origin

def write_wad(wad, maps):
    for map_obj in maps:
        map_data = omg.MapEditor()
        map_data.vertexes = [omg.Vertex(x=v.x, y=v.y) for v in map_obj.vertices]
        map_data.linedefs = [
            omg.Linedef(vx_a=ld.v0, vx_b=ld.v1, flags=ld.flags, action=ld.special, 
                        tag=ld.tag, front=ld.side0, back=ld.side1)
            for ld in map_obj.linedefs
        ]
        map_data.sidedefs = [
            omg.Sidedef(off_x=sd.x_offset, off_y=sd.y_offset, tx_up=sd.upper,
                        tx_low=sd.lower, tx_mid=sd.middle, sector=sd.sector)
            for sd in map_obj.sidedefs
        ]
        map_data.sectors = [
            omg.Sector(z_floor=s.floor_height, z_ceil=s.ceiling_height,
                       tx_floor=s.floor_texture, tx_ceil=s.ceiling_texture,
                       light=s.light_level, type=s.type, tag=s.tag)
            for s in map_obj.sectors
        ]
        map_data.things = [
            omg.Thing(x=t.x_pos, y=t.y_pos, angle=t.angle, type=t.type, flags=t.flags)
            for t in map_obj.things
        ]
        wad.maps[map_obj.name] = map_data.to_lumps()

def copy_wad_resources(wad):
    input_wad = omg.WAD("./wads/doom.wad")  # Load original WAD
    wad.sprites = input_wad.sprites
    wad.patches = input_wad.patches
    wad.flats = input_wad.flats
    wad.colormaps = input_wad.colormaps
    wad.ztextures = input_wad.ztextures
    wad.glmaps = input_wad.glmaps
    wad.udmfmaps = input_wad.udmfmaps
    wad.music = input_wad.music
    wad.sounds = input_wad.sounds
    wad.txdefs = input_wad.txdefs
    wad.graphics = input_wad.graphics
    wad.data = input_wad.data

maps = []
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".wad"):
        wad_path = os.path.join(input_dir, filename)
        wad = omg.WAD(wad_path)
        for lump in wad.maps:
            map_data = wad.maps[lump]
            things = read_things(map_data)
            origin = read_origin(things)
            maps.append(Map(
                name=lump,
                vertices=read_vertices(map_data),
                linedefs=read_linedefs(map_data),
                sidedefs=read_sidedefs(map_data),
                sectors=read_sectors(map_data),
                things=things,
                origin=origin
            ))

print(len(maps))

wad = omg.WAD()
write_wad(wad, maps[:2])
copy_wad_resources(wad);
wad.to_file("./output.wad")
