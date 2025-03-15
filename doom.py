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

maps = []
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".wad"):
        wad_path = os.path.join(input_dir, filename)
        wad = omg.WAD(wad_path)
        for lump in wad.maps:
            map_data = wad.maps[lump]
            maps.append(Map(
                name=lump,
                vertices=read_vertices(map_data),
                linedefs=read_linedefs(map_data),
                sidedefs=read_sidedefs(map_data),
                sectors=read_sectors(map_data),
                things=read_things(map_data)
            ))
