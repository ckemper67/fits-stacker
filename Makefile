CXX      ?= c++
CXXFLAGS  = -std=c++17 -O2 -march=native -I donuts

# pkg-config probes (works on Linux and macOS with pkgconf/homebrew)
CFITSIO_CFLAGS := $(shell pkg-config --cflags cfitsio 2>/dev/null)
CFITSIO_LIBS   := $(shell pkg-config --libs   cfitsio 2>/dev/null)
PNG_CFLAGS     := $(shell pkg-config --cflags libpng  2>/dev/null)
PNG_LIBS       := $(shell pkg-config --libs   libpng  2>/dev/null)

# Fallback: homebrew prefix (macOS; silent if brew not present)
BREW_PREFIX := $(shell brew --prefix 2>/dev/null)
ifneq ($(BREW_PREFIX),)
    CXXFLAGS += -I$(BREW_PREFIX)/include
    LDFLAGS  += -L$(BREW_PREFIX)/lib
    ifeq ($(CFITSIO_LIBS),)
        CFITSIO_LIBS = -lcfitsio
    endif
    ifeq ($(PNG_LIBS),)
        PNG_LIBS = -lpng
    endif
endif

# Last-resort fallback: link by name (works when libs are on the default path)
ifeq ($(CFITSIO_LIBS),)
    CFITSIO_LIBS = -lcfitsio
endif
ifeq ($(PNG_LIBS),)
    PNG_LIBS = -lpng
endif

# Optional OpenCV backend for warp and debayer: make OPENCV=1
OPENCV ?= 0
ifeq ($(OPENCV),1)
    OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null || \
                              pkg-config --cflags opencv  2>/dev/null)
    OPENCV_LIBS   := $(shell pkg-config --libs   opencv4 2>/dev/null || \
                              pkg-config --libs   opencv  2>/dev/null)
    ifeq ($(OPENCV_LIBS),)
        $(error OpenCV not found via pkg-config. Install opencv4 or set OPENCV_CFLAGS/OPENCV_LIBS manually.)
    endif
    CXXFLAGS += -DHAVE_OPENCV $(OPENCV_CFLAGS)
    LDFLAGS  += $(OPENCV_LIBS)
endif

SRCS = fits_stack.cpp donuts/donuts.cpp
BIN  = fits_stack

.PHONY: all clean

all: $(BIN)

$(BIN): $(SRCS)
	$(CXX) $(CXXFLAGS) $(CFITSIO_CFLAGS) $(PNG_CFLAGS) \
	    $^ \
	    $(LDFLAGS) $(CFITSIO_LIBS) $(PNG_LIBS) \
	    -o $@

clean:
	rm -f $(BIN)
