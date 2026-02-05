#!/bin/bash
#
# Build script for DeepSeek-OCR Singularity container
# This script clears caches and temporary files before building
#
# Usage:
#   ./build.sh                    # Build with default settings
#   ./build.sh --force            # Force rebuild even if .sif exists
#   ./build.sh --no-cache         # Clear all Singularity cache before build
#   ./build.sh --tmpdir /path     # Use custom temp directory
#   ./build.sh --fakeroot         # Build with fakeroot (no sudo required if configured)
#

set -e

# Default settings
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEF_FILE="${SCRIPT_DIR}/deepseek_ocr.def"
SIF_FILE="${SCRIPT_DIR}/deepseek_ocr.sif"
FORCE_BUILD=false
CLEAR_CACHE=false
USE_FAKEROOT=false
CUSTOM_TMPDIR=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    echo "DeepSeek-OCR Singularity Container Build Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -f, --force       Force rebuild even if .sif file exists"
    echo "  -c, --no-cache    Clear all Singularity cache before building"
    echo "  -t, --tmpdir DIR  Use custom temporary directory for build"
    echo "  --fakeroot        Use fakeroot instead of sudo (requires admin setup)"
    echo ""
    echo "Examples:"
    echo "  $0                           # Standard build"
    echo "  $0 --force --no-cache        # Clean rebuild"
    echo "  $0 --tmpdir /scratch/tmp     # Use /scratch/tmp for build files"
    echo "  $0 --fakeroot                # Build without sudo"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -f|--force)
            FORCE_BUILD=true
            shift
            ;;
        -c|--no-cache)
            CLEAR_CACHE=true
            shift
            ;;
        -t|--tmpdir)
            CUSTOM_TMPDIR="$2"
            shift 2
            ;;
        --fakeroot)
            USE_FAKEROOT=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if singularity is installed
if ! command -v singularity &> /dev/null; then
    print_error "Singularity is not installed or not in PATH"
    exit 1
fi

print_info "Singularity version: $(singularity --version)"

# Check if definition file exists
if [[ ! -f "${DEF_FILE}" ]]; then
    print_error "Definition file not found: ${DEF_FILE}"
    exit 1
fi

# Check if .sif already exists
if [[ -f "${SIF_FILE}" ]] && [[ "${FORCE_BUILD}" == false ]]; then
    print_warning "Container already exists: ${SIF_FILE}"
    read -p "Do you want to rebuild? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Build cancelled"
        exit 0
    fi
    FORCE_BUILD=true
fi

# Clear Singularity cache if requested
if [[ "${CLEAR_CACHE}" == true ]]; then
    print_info "Clearing Singularity cache..."

    # Clear user cache
    if [[ -d "${HOME}/.singularity/cache" ]]; then
        rm -rf "${HOME}/.singularity/cache"/*
        print_info "Cleared user cache: ${HOME}/.singularity/cache"
    fi

    # Clear system cache (requires sudo)
    if [[ "${USE_FAKEROOT}" == false ]]; then
        if [[ -d "/var/singularity/cache" ]]; then
            sudo rm -rf /var/singularity/cache/* 2>/dev/null || true
            print_info "Cleared system cache: /var/singularity/cache"
        fi
    fi

    # Run singularity cache clean
    singularity cache clean -f 2>/dev/null || true
    print_success "Cache cleared"
fi

# Set up temporary directory
if [[ -n "${CUSTOM_TMPDIR}" ]]; then
    if [[ ! -d "${CUSTOM_TMPDIR}" ]]; then
        print_info "Creating temp directory: ${CUSTOM_TMPDIR}"
        mkdir -p "${CUSTOM_TMPDIR}"
    fi
    export SINGULARITY_TMPDIR="${CUSTOM_TMPDIR}"
    export TMPDIR="${CUSTOM_TMPDIR}"
    print_info "Using temp directory: ${CUSTOM_TMPDIR}"
else
    # Create a local tmp directory to avoid /tmp space issues
    LOCAL_TMP="${SCRIPT_DIR}/tmp_build"
    mkdir -p "${LOCAL_TMP}"
    export SINGULARITY_TMPDIR="${LOCAL_TMP}"
    export TMPDIR="${LOCAL_TMP}"
    print_info "Using temp directory: ${LOCAL_TMP}"
fi

# Clean up old tmp files
print_info "Cleaning up old temporary files..."
rm -rf "${SINGULARITY_TMPDIR:?}"/* 2>/dev/null || true

# Remove old .sif if force rebuild
if [[ "${FORCE_BUILD}" == true ]] && [[ -f "${SIF_FILE}" ]]; then
    print_info "Removing existing container: ${SIF_FILE}"
    rm -f "${SIF_FILE}"
fi

# Build the container
print_info "Starting Singularity build..."
print_info "Definition file: ${DEF_FILE}"
print_info "Output file: ${SIF_FILE}"
echo ""

BUILD_CMD="singularity build"

if [[ "${USE_FAKEROOT}" == true ]]; then
    BUILD_CMD="${BUILD_CMD} --fakeroot"
    print_info "Building with fakeroot..."
else
    BUILD_CMD="sudo -E ${BUILD_CMD}"
    print_info "Building with sudo..."
fi

# Run build
START_TIME=$(date +%s)

if ${BUILD_CMD} "${SIF_FILE}" "${DEF_FILE}"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MINUTES=$((DURATION / 60))
    SECONDS=$((DURATION % 60))

    print_success "Build completed successfully!"
    print_info "Build time: ${MINUTES}m ${SECONDS}s"
    print_info "Container: ${SIF_FILE}"
    print_info "Size: $(du -h "${SIF_FILE}" | cut -f1)"
    echo ""

    # Clean up temp directory
    print_info "Cleaning up temporary files..."
    rm -rf "${SINGULARITY_TMPDIR:?}"/* 2>/dev/null || true

    # Show usage
    echo ""
    echo "=========================================="
    echo "Container built successfully!"
    echo "=========================================="
    echo ""
    echo "Quick start commands:"
    echo ""
    echo "  # Interactive shell:"
    echo "  singularity shell --nv ${SIF_FILE}"
    echo ""
    echo "  # Run training:"
    echo "  singularity exec --nv ${SIF_FILE} \\"
    echo "      python /opt/torchtitan/torchtitan/train.py \\"
    echo "      --config /opt/torchtitan/torchtitan/models/deepseek_ocr/train_configs/debug.toml"
    echo ""
    echo "  # Multi-GPU training:"
    echo "  singularity exec --nv ${SIF_FILE} \\"
    echo "      torchrun --nproc_per_node=4 \\"
    echo "      /opt/torchtitan/torchtitan/train.py \\"
    echo "      --config /opt/torchtitan/torchtitan/models/deepseek_ocr/train_configs/debug.toml"
    echo ""
else
    print_error "Build failed!"
    print_info "Check the output above for errors"
    print_info "Try running with --no-cache to clear cached layers"

    # Clean up on failure
    rm -rf "${SINGULARITY_TMPDIR:?}"/* 2>/dev/null || true

    exit 1
fi
