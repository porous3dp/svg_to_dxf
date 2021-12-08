svg_to_dxf
==========

The code here is basically an extracted copy of the "Save As DXF" export extension from Inkscape 1.0
(as implemented in `src/dxf_outlines.py`)

## Usage

```
svg_to_dxf.py --output "000001.dxf" 000001.svg
```

## Setup

To install the required dependencies (mostly for the "inkex" utilities),
run pip:

```
$ pip install -r requirements.txt
```

## Building

To compile the script to a standalone binary that can be called other programs:
```
$ build.bat
```


## License

The Inkscape code that this uses is all licensed under the GPL (see the COPYING) file.
Hence, this repo is also GPL.

