# ============================================================================
# Monte Carlo Option Pricer — C Version
# ============================================================================
# Build system for compiling the Monte Carlo pricing engine from scratch
# using only the C standard library (libc + libm). No external dependencies.
#
# Design Constraints Met:
# - Maximum 5 functions per source file
# - Maximum 25 lines per function
# - All structures use static memory allocation
# - No external library dependencies beyond standard C
# - C11 standard for portability and modern features
#
# Usage:
#   make          - Compile the project
#   make run      - Compile and execute pricing pipeline
#   make test     - Compile and run test suite
#   make clean    - Remove object files and executable
#   make distclean - Remove all generated files including outputs
#   make help     - Display this help message
#
# Author: Claudia Maria Lopez Bombin
# License: MIT
# Repository: github.com/claudialbombin/monte-carlo-option-pricer
# ============================================================================

# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c11 -Iinclude
LDFLAGS = -lm

# Directory structure
SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj
BIN_DIR = bin

# All source files (each with ≤5 functions)
SOURCES = $(SRC_DIR)/rng.c \
          $(SRC_DIR)/models.c \
          $(SRC_DIR)/options.c \
          $(SRC_DIR)/greeks.c \
          $(SRC_DIR)/convergence.c \
          $(SRC_DIR)/main.c

# Generate object file paths from source file paths
OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SOURCES))

# Final executable
TARGET = $(BIN_DIR)/mc_option_pricer

# ============================================================================
# Primary targets
# ============================================================================

# Default: compile everything
all: dirs $(TARGET)
	@echo ""
	@echo "Build complete: $(TARGET)"
	@echo "Run with: make run"
	@echo ""

# Create necessary directories
dirs:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(BIN_DIR)
	@mkdir -p data

# Link all object files into final executable
$(TARGET): $(OBJECTS)
	@echo "Linking $(TARGET)..."
	$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)
	@echo "Executable created: $(TARGET)"

# Compile each source file into object file
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -c $< -o $@

# ============================================================================
# Execution targets
# ============================================================================

# Run complete pricing pipeline
run: $(TARGET)
	@echo ""
	@echo "Running Monte Carlo Option Pricer..."
	@echo ""
	./$(TARGET)

# Compile and run test suite
test: CFLAGS += -DTEST_MODE
test: clean all
	@echo ""
	@echo "Running test suite..."
	@echo ""
	./$(TARGET) --test

# Run with verbose output
verbose: CFLAGS += -DVERBOSE_MODE
verbose: clean all
	@echo ""
	@echo "Running in VERBOSE mode..."
	@echo ""
	./$(TARGET)

# ============================================================================
# Utility targets
# ============================================================================

# Remove object files and executable
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(OBJ_DIR)
	rm -rf $(BIN_DIR)
	@echo "Clean complete."

# Remove everything including generated outputs
distclean: clean
	@echo "Removing all generated files..."
	rm -f data/convergence_N.csv
	rm -f data/convergence_dt.csv
	rm -f data/greeks_convergence.csv
	@echo "Distclean complete."

# Display help
help:
	@echo ""
	@echo "Monte Carlo Option Pricer - Build System"
	@echo "========================================="
	@echo ""
	@echo "BUILD TARGETS:"
	@echo "  make          - Compile the project"
	@echo "  make all      - Same as make (compile everything)"
	@echo ""
	@echo "EXECUTION TARGETS:"
	@echo "  make run      - Run complete pricing pipeline"
	@echo "  make test     - Run test suite"
	@echo "  make verbose  - Run with detailed output"
	@echo ""
	@echo "CLEANUP TARGETS:"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make distclean - Remove everything generated"
	@echo ""
	@echo "UTILITY TARGETS:"
	@echo "  make help     - Show this message"
	@echo ""

# ============================================================================
# Phony targets (not real files)
# ============================================================================

.PHONY: all dirs run test verbose clean distclean help