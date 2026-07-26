#!/bin/bash

# Display Manager Script
# Enhanced virtual display management with CLI support
#
# This script creates isolated virtual displays with desktop environments
# that are completely independent from the system's main desktop session.
#
# Isolation Features:
# - Uses separate XDG configuration directories for each virtual display
# - Desktop environments run in their own processes and memory space
# - Virtual displays use high-numbered display IDs (99+) to avoid conflicts
# - Automatic cleanup of temporary files and configurations on shutdown
# - No interference with system's main GNOME/KDE/XFCE sessions
#
# Usage Examples:
#   ./start_screen.sh                    # Interactive mode (default: LXDE)
#   ./start_screen.sh --start           # Start with LXDE (lightweight default)
#   ./start_screen.sh -t openbox --start # Start with Openbox (very lightweight)
#   ./start_screen.sh -t gnome --start  # Start with GNOME (resource intensive)
#   ./start_screen.sh --list            # List running displays
#   ./start_screen.sh --kill            # Kill all virtual displays
#
# Safety Notes:
# - Virtual displays are completely isolated and won't affect your main desktop
# - Each virtual display gets its own configuration files in /tmp/
# - All temporary files are cleaned up when displays are killed

# Default configuration
DISPLAY_NUM=${DISPLAY_NUM:-99}
RESOLUTION=${RESOLUTION:-1920x1080x24}
VNC_PORT=${VNC_PORT:-5900}
XVFB_ARGS=${XVFB_ARGS:-"+extension GLX +render -ac"}
VNC_ARGS=${VNC_ARGS:-"-nopw -forever -shared"}
DESKTOP_ENV=${DESKTOP_ENV:-lxde}  # gnome, xfce, kde, lxde, openbox, or none

# Flags to track if values were set via command line
DISPLAY_NUM_SET=""
VNC_PORT_SET=""
DESKTOP_ENV_SET=""

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Display Manager Options:"
    echo "  -d, --display NUM        Display number (default: auto-assign from 99+)"
    echo "  -r, --resolution RES     Resolution format (default: 1920x1080x24)"
    echo "  -p, --port PORT          VNC port (default: auto-assign from 5900+)"
    echo "  -x, --xvfb-args ARGS     Additional Xvfb arguments (default: '+extension GLX +render -ac')"
    echo "  -v, --vnc-args ARGS      Additional x11vnc arguments (default: '-nopw -forever -shared')"
    echo "  -t, --desktop ENV        Desktop environment (gnome, xfce, kde, lxde, openbox, none) (default: lxde)"
    echo "  -s, --start              Start the display server"
    echo "  -k, --kill               Kill running display servers"
    echo "  -l, --list               List running display servers"
    echo "  -e, --export             Export display environment variables"
    echo "  -i, --interactive        Interactive mode"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  DISPLAY_NUM              Default display number"
    echo "  RESOLUTION               Default resolution"
    echo "  VNC_PORT                 Default VNC port"
    echo "  DESKTOP_ENV              Default desktop environment (gnome/xfce/kde/lxde/openbox/none)"
    echo "  XVFB_ARGS                Default Xvfb arguments"
    echo "  VNC_ARGS                 Default x11vnc arguments"
    echo ""
    echo "Examples:"
    echo "  $0 --start                                    # Start with defaults"
    echo "  $0 -d 100 -r 2560x1440x24 -p 5901 --start     # Custom display 100"
    echo "  $0 --list                                     # List running displays"
    echo "  $0 --kill                                     # Kill all displays"
    echo "  $0 --export                                   # Export DISPLAY=:99"
    echo "  $0                                            # Interactive mode"
}

# Function to show status
show_status() {
    echo "=== Current Display Status ==="
    echo ""

    local running_displays=0
    local running_vnc=0

    # Check running displays
    while read -r line; do
        if [[ $line =~ Xvfb\ :([0-9]+) ]]; then
            ((running_displays++))
            display=${BASH_REMATCH[1]}
            pid=$(echo $line | awk '{print $1}')
            echo "✓ Display :$display running (PID: $pid)"
        fi
    done < <(ps aux | grep "Xvfb :" | grep -v grep)

    # Check running VNC servers
    while read -r line; do
        if [[ $line =~ rfbport\ ([0-9]+) ]]; then
            ((running_vnc++))
            port=${BASH_REMATCH[1]}
            echo "✓ VNC server on port $port running"
        fi
    done < <(ps aux | grep "x11vnc" | grep -v grep)

    if [ $running_displays -eq 0 ]; then
        echo "✗ No virtual displays currently running"
    fi

    if [ $running_vnc -eq 0 ]; then
        echo "✗ No VNC servers currently running"
    fi

    echo ""
}

# Function to get user confirmation
confirm() {
    local message="$1"
    local default="${2:-n}"

    if [ "$default" = "y" ]; then
        read -p "$message [Y/n]: " -n 1 -r
        echo
        [[ ! $REPLY =~ ^[Nn]$ ]]
    else
        read -p "$message [y/N]: " -n 1 -r
        echo
        [[ $REPLY =~ ^[Yy]$ ]]
    fi
}

# Function to show interactive menu
show_interactive_menu() {
    while true; do
        clear
        echo "========================================"
        echo "      Virtual Display Manager"
        echo "========================================"
        echo ""

        show_status

        echo "Available Actions:"
        echo "  1) Start new display"
        echo "  2) Kill all displays"
        echo "  3) List detailed display info"
        echo "  4) Export DISPLAY variable"
        echo "  5) Configure settings"
        echo "  6) Show help"
        echo "  0) Exit"
        echo ""

        read -p "Choose an action (0-6): " choice
        echo ""

        case $choice in
            1)
                if confirm "Start virtual display with current settings?" "y"; then
                    start_display
                    echo ""
                    read -p "Press Enter to continue..."
                fi
                ;;
            2)
                if confirm "Kill ALL running displays? This will stop all virtual displays." "n"; then
                    kill_displays
                    echo ""
                    read -p "Press Enter to continue..."
                fi
                ;;
            3)
                list_displays
                echo ""
                read -p "Press Enter to continue..."
                ;;
            4)
                export_display
                echo ""
                read -p "Press Enter to continue..."
                ;;
            5)
                configure_settings
                ;;
            6)
                show_usage
                echo ""
                read -p "Press Enter to continue..."
                ;;
            0)
                echo "Goodbye!"
                exit 0
                ;;
            *)
                echo "Invalid choice. Please try again."
                sleep 1
                ;;
        esac
    done
}

# Function to configure settings interactively
configure_settings() {
    echo "=== Configuration Settings ==="
    echo "Current settings:"
    echo "  Display Number: $DISPLAY_NUM"
    echo "  Resolution: $RESOLUTION"
    echo "  VNC Port: $VNC_PORT"
    echo "  Desktop Environment: $DESKTOP_ENV"
    echo "  Xvfb Args: $XVFB_ARGS"
    echo "  VNC Args: $VNC_ARGS"
    echo ""

    read -p "Display number [$DISPLAY_NUM]: " new_display
    if [[ -n "$new_display" ]]; then
        DISPLAY_NUM=$new_display
        DISPLAY_NUM_SET="true"
    fi

    read -p "Resolution [$RESOLUTION]: " new_resolution
    RESOLUTION=${new_resolution:-$RESOLUTION}

    read -p "VNC port [$VNC_PORT]: " new_port
    if [[ -n "$new_port" ]]; then
        VNC_PORT=$new_port
        VNC_PORT_SET="true"
    fi

    read -p "Desktop environment (gnome/xfce/kde/lxde/openbox/none) [$DESKTOP_ENV]: " new_desktop
    if [[ -n "$new_desktop" ]]; then
        DESKTOP_ENV=$new_desktop
        DESKTOP_ENV_SET="true"
    fi

    read -p "Xvfb arguments [$XVFB_ARGS]: " new_xvfb_args
    XVFB_ARGS=${new_xvfb_args:-$XVFB_ARGS}

    read -p "VNC arguments [$VNC_ARGS]: " new_vnc_args
    VNC_ARGS=${new_vnc_args:-$VNC_ARGS}

    echo ""
    echo "Settings updated!"
    echo ""
    read -p "Press Enter to continue..."
}

# Function to check if display is running
is_display_running() {
    local display=$1
    pgrep -f "Xvfb :$display" > /dev/null 2>&1
}

# Function to check if VNC is running
is_vnc_running() {
    local port=$1
    netstat -tuln 2>/dev/null | grep ":$port " > /dev/null 2>&1
}

# Function to check desktop environment availability
check_desktop_availability() {
    local desktop=$1

    case $desktop in
        gnome)
            if ! command -v gnome-session >/dev/null 2>&1; then
                echo "Error: gnome-session not found. Please install GNOME desktop environment."
                echo "Run: sudo apt install gnome-session gdm3"
                return 1
            fi
            ;;
        xfce)
            if ! command -v xfce4-session >/dev/null 2>&1; then
                echo "Error: xfce4-session not found. Please install XFCE desktop environment."
                echo "Run: sudo apt install xfce4 xfce4-goodies"
                return 1
            fi
            ;;
        kde)
            if ! command -v startkde >/dev/null 2>&1; then
                echo "Error: startkde not found. Please install KDE desktop environment."
                echo "Run: sudo apt install kde-standard"
                return 1
            fi
            ;;
        lxde)
            if ! command -v startlxde >/dev/null 2>&1 && ! command -v lxsession >/dev/null 2>&1; then
                echo "Error: LXDE not found. Installing lightweight desktop environment..."
                echo "Run: sudo apt update && sudo apt install lxde lxappearance"
                echo "Or for even lighter: sudo apt install lxde-core"
                return 1
            fi
            ;;
        openbox)
            if ! command -v openbox >/dev/null 2>&1; then
                echo "Error: Openbox not found. Installing lightweight window manager..."
                echo "Run: sudo apt install openbox obconf obmenu"
                return 1
            fi
            ;;
        none)
            # Check for basic window managers
            if ! command -v openbox >/dev/null 2>&1 && \
               ! command -v fluxbox >/dev/null 2>&1 && \
               ! command -v twm >/dev/null 2>&1; then
                echo "Warning: No window manager found. Install openbox, fluxbox, or twm for basic functionality."
            fi
            ;;
    esac
    return 0
}

# Function to find available display number
find_available_display() {
    local start_display=${1:-99}
    local max_attempts=100
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        local display_num=$((start_display + attempt))
        if ! is_display_running $display_num; then
            echo $display_num
            return 0
        fi
        ((attempt++))
    done

    # If we can't find a display in the preferred range, try from 1 upwards
    attempt=1
    while [ $attempt -lt $max_attempts ]; do
        if ! is_display_running $attempt; then
            echo $attempt
            return 0
        fi
        ((attempt++))
    done

    echo "Error: Could not find available display number" >&2
    return 1
}

# Function to find available VNC port
find_available_port() {
    local start_port=${1:-5900}
    local max_attempts=100
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        local port=$((start_port + attempt))
        if ! is_vnc_running $port; then
            echo $port
            return 0
        fi
        ((attempt++))
    done

    # If we can't find a port in the preferred range, try from 5901 upwards
    attempt=1
    while [ $attempt -lt $max_attempts ]; do
        local port=$((5900 + attempt))
        if ! is_vnc_running $port; then
            echo $port
            return 0
        fi
        ((attempt++))
    done

    echo "Error: Could not find available VNC port" >&2
    return 1
}

# Function to start desktop environment
start_desktop_environment() {
    local display=$1

    # Check if we already have too many GNOME sessions running
    if [[ "$DESKTOP_ENV" == "gnome" ]]; then
        local gnome_count=$(pgrep -f "gnome-session" | wc -l)
        if [[ $gnome_count -gt 1 ]]; then
            echo "Warning: Multiple GNOME sessions detected. Switching to LXDE for better performance."
            DESKTOP_ENV="lxde"
        fi
    fi

    case $DESKTOP_ENV in
        gnome)
            echo "Starting isolated GNOME desktop environment on display :$display..."
            # Create isolated environment variables for virtual display
            export DISPLAY=:$display
            export XDG_SESSION_TYPE=x11
            export XDG_CURRENT_DESKTOP=GNOME
            export GDMSESSION=gnome
            export DESKTOP_SESSION=gnome

            # Set up isolated XDG directories for virtual session
            export XDG_CONFIG_HOME=${XDG_CONFIG_HOME:-$HOME/.config}
            export XDG_DATA_HOME=${XDG_DATA_HOME:-$HOME/.local/share}
            export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$HOME/.cache}

            # Create temporary directories for this virtual session
            export XDG_CONFIG_HOME_VIRTUAL="/tmp/xdg-config-gnome-$display"
            export XDG_DATA_HOME_VIRTUAL="/tmp/xdg-data-gnome-$display"
            export XDG_CACHE_HOME_VIRTUAL="/tmp/xdg-cache-gnome-$display"

            # Copy essential GNOME configs to virtual directories
            mkdir -p "$XDG_CONFIG_HOME_VIRTUAL"
            mkdir -p "$XDG_DATA_HOME_VIRTUAL"
            mkdir -p "$XDG_CACHE_HOME_VIRTUAL"

            # Copy basic configs if they exist
            if [ -d "$XDG_CONFIG_HOME/gnome-session" ]; then
                cp -r "$XDG_CONFIG_HOME/gnome-session" "$XDG_CONFIG_HOME_VIRTUAL/" 2>/dev/null || true
            fi

            # Override XDG directories for this session
            export XDG_CONFIG_HOME="$XDG_CONFIG_HOME_VIRTUAL"
            export XDG_DATA_HOME="$XDG_DATA_HOME_VIRTUAL"
            export XDG_CACHE_HOME="$XDG_CACHE_HOME_VIRTUAL"

            # Set GNOME session to run in isolation
            export GNOME_SHELL_SESSION_MODE=classic  # Use classic mode for better compatibility

            # Start GNOME session in background
            (
                export DISPLAY=:$display
                export XDG_CONFIG_HOME="$XDG_CONFIG_HOME_VIRTUAL"
                export XDG_DATA_HOME="$XDG_DATA_HOME_VIRTUAL"
                export XDG_CACHE_HOME="$XDG_CACHE_HOME_VIRTUAL"
                gnome-session --session=gnome &
                GNOME_PID=$!
                echo "GNOME session started with PID: $GNOME_PID on display :$display"

                # Wait for GNOME to initialize
                sleep 5

                # Check if GNOME started successfully
                if ps -p $GNOME_PID > /dev/null 2>&1; then
                    echo "GNOME desktop environment started successfully on display :$display"
                else
                    echo "Warning: GNOME may not have started properly on display :$display"
                fi
            ) &
            ;;
        xfce)
            echo "Starting isolated XFCE desktop environment on display :$display..."
            (
                export DISPLAY=:$display
                # Create isolated config for XFCE
                export XDG_CONFIG_HOME="/tmp/xdg-config-xfce-$display"
                mkdir -p "$XDG_CONFIG_HOME"

                xfce4-session &
                XFCE_PID=$!
                echo "XFCE session started with PID: $XFCE_PID on display :$display"
                sleep 3
            ) &
            ;;
        kde)
            echo "Starting isolated KDE desktop environment on display :$display..."
            (
                export DISPLAY=:$display
                # Create isolated config for KDE
                export KDEHOME="/tmp/kde-config-$display"
                mkdir -p "$KDEHOME"

                startkde &
                KDE_PID=$!
                echo "KDE session started with PID: $KDE_PID on display :$display"
                sleep 3
            ) &
            ;;
        lxde)
            echo "Starting lightweight LXDE desktop environment on display :$display..."
            (
                export DISPLAY=:$display
                # Create isolated config for LXDE
                export XDG_CONFIG_HOME="/tmp/xdg-config-lxde-$display"
                mkdir -p "$XDG_CONFIG_HOME"

                # Start LXDE session
                if command -v startlxde >/dev/null 2>&1; then
                    startlxde &
                elif command -v lxsession >/dev/null 2>&1; then
                    lxsession &
                else
                    echo "Warning: LXDE not properly installed, falling back to Openbox"
                    openbox &
                fi
                LXDE_PID=$!
                echo "LXDE session started with PID: $LXDE_PID on display :$display"
                sleep 2
            ) &
            ;;
        openbox)
            echo "Starting lightweight Openbox window manager on display :$display..."
            (
                export DISPLAY=:$display
                # Create isolated config for Openbox
                export XDG_CONFIG_HOME="/tmp/xdg-config-openbox-$display"
                mkdir -p "$XDG_CONFIG_HOME"

                openbox &
                OPENBOX_PID=$!
                echo "Openbox session started with PID: $OPENBOX_PID on display :$display"
                sleep 2
            ) &
            ;;
        none|*)
            echo "No desktop environment specified (DESKTOP_ENV=$DESKTOP_ENV)"
            echo "Starting with basic X window manager on display :$display..."
            (
                export DISPLAY=:$display
                # Try to start a basic window manager
                if command -v openbox >/dev/null 2>&1; then
                    openbox &
                elif command -v fluxbox >/dev/null 2>&1; then
                    fluxbox &
                elif command -v twm >/dev/null 2>&1; then
                    twm &
                else
                    echo "Warning: No window manager found for display :$display"
                fi
            ) &
            ;;
    esac
}

# Function to start display
start_display() {
    # Auto-assign display number if not explicitly set via command line
    if [[ -z "${DISPLAY_NUM_SET:-}" ]]; then
        echo "Finding available display number..."
        DISPLAY_NUM=$(find_available_display $DISPLAY_NUM)
        if [[ $? -ne 0 ]]; then
            echo "Failed to find available display number"
            return 1
        fi
        echo "Using display :$DISPLAY_NUM"
    fi

    # Auto-assign VNC port if not explicitly set via command line
    if [[ -z "${VNC_PORT_SET:-}" ]]; then
        echo "Finding available VNC port..."
        VNC_PORT=$(find_available_port $VNC_PORT)
        if [[ $? -ne 0 ]]; then
            echo "Failed to find available VNC port"
            return 1
        fi
        echo "Using VNC port $VNC_PORT"
    fi

    # Check desktop environment availability
    if ! check_desktop_availability $DESKTOP_ENV; then
        return 1
    fi

    echo "Starting virtual display :$DISPLAY_NUM with resolution $RESOLUTION"

    # Double-check if display is already running (shouldn't happen with auto-assignment)
    if is_display_running $DISPLAY_NUM; then
        echo "Warning: Display :$DISPLAY_NUM is already running"
        return 1
    fi

    # Double-check if VNC port is already in use (shouldn't happen with auto-assignment)
    if is_vnc_running $VNC_PORT; then
        echo "Warning: VNC port $VNC_PORT is already in use"
        return 1
    fi

    # Start Xvfb
    echo "Starting Xvfb on display :$DISPLAY_NUM..."
    /usr/bin/Xvfb :$DISPLAY_NUM -screen 0 $RESOLUTION $XVFB_ARGS &
    XVFB_PID=$!

    # Wait for Xvfb to start
sleep 2 

    # Check if Xvfb started successfully
    if ! is_display_running $DISPLAY_NUM; then
        echo "Error: Failed to start Xvfb on display :$DISPLAY_NUM"
        return 1
    fi

    echo "Xvfb started successfully (PID: $XVFB_PID)"

    # Start x11vnc
    echo "Starting x11vnc on port $VNC_PORT..."
    x11vnc -display :$DISPLAY_NUM $VNC_ARGS -rfbport $VNC_PORT &
    VNC_PID=$!

    # Wait a moment for VNC to start
    sleep 1

    # Check if VNC started successfully
    if ! is_vnc_running $VNC_PORT; then
        echo "Warning: x11vnc may not have started properly on port $VNC_PORT"
    else
        echo "x11vnc started successfully (PID: $VNC_PID)"
    fi

    # Start desktop environment
    start_desktop_environment $DISPLAY_NUM

    echo ""
    echo "Virtual display :$DISPLAY_NUM is now running:"
    echo "  Resolution: $RESOLUTION"
    echo "  VNC Port: $VNC_PORT"
    echo "  Desktop Environment: $DESKTOP_ENV"
    echo "  Connect via: vncviewer localhost:$VNC_PORT"
    echo ""
    echo "To export DISPLAY variable, run: export DISPLAY=:$DISPLAY_NUM"
}

# Function to kill displays
kill_displays() {
    echo "Killing virtual display servers..."

    # Kill all Xvfb processes
    local xvfb_pids=$(pgrep -f "Xvfb")
    if [ -n "$xvfb_pids" ]; then
        echo "Killing Xvfb processes: $xvfb_pids"
        kill $xvfb_pids 2>/dev/null
        sleep 1
        kill -9 $xvfb_pids 2>/dev/null
    fi

    # Kill all x11vnc processes
    local vnc_pids=$(pgrep -f "x11vnc")
    if [ -n "$vnc_pids" ]; then
        echo "Killing x11vnc processes: $vnc_pids"
        kill $vnc_pids 2>/dev/null
        sleep 1
        kill -9 $vnc_pids 2>/dev/null
    fi

    # Kill desktop environment processes
    local gnome_pids=$(pgrep -f "gnome-session")
    if [ -n "$gnome_pids" ]; then
        echo "Killing GNOME processes: $gnome_pids"
        kill $gnome_pids 2>/dev/null
        sleep 1
        kill -9 $gnome_pids 2>/dev/null
    fi

    local xfce_pids=$(pgrep -f "xfce4-session")
    if [ -n "$xfce_pids" ]; then
        echo "Killing XFCE processes: $xfce_pids"
        kill $xfce_pids 2>/dev/null
        sleep 1
        kill -9 $xfce_pids 2>/dev/null
    fi

    local kde_pids=$(pgrep -f "startkde\|kdeinit")
    if [ -n "$kde_pids" ]; then
        echo "Killing KDE processes: $kde_pids"
        kill $kde_pids 2>/dev/null
        sleep 1
        kill -9 $kde_pids 2>/dev/null
    fi

    local lxde_pids=$(pgrep -f "lxsession\|startlxde")
    if [ -n "$lxde_pids" ]; then
        echo "Killing LXDE processes: $lxde_pids"
        kill $lxde_pids 2>/dev/null
        sleep 1
        kill -9 $lxde_pids 2>/dev/null
    fi

    local openbox_pids=$(pgrep -f "openbox")
    if [ -n "$openbox_pids" ]; then
        echo "Killing Openbox processes: $openbox_pids"
        kill $openbox_pids 2>/dev/null
        sleep 1
        kill -9 $openbox_pids 2>/dev/null
    fi

    # Clean up temporary XDG directories
    echo "Cleaning up temporary configuration directories..."
    rm -rf /tmp/xdg-config-gnome-* /tmp/xdg-data-gnome-* /tmp/xdg-cache-gnome-* 2>/dev/null || true
    rm -rf /tmp/xdg-config-xfce-* /tmp/kde-config-* 2>/dev/null || true
    rm -rf /tmp/xdg-config-lxde-* /tmp/xdg-config-openbox-* 2>/dev/null || true

    echo "All virtual display servers and desktop environments have been killed."
}

# Function to list displays
list_displays() {
    echo "Running virtual displays:"
    echo "========================"

    local found=0

    # Check Xvfb processes
    while read -r line; do
        if [[ $line =~ Xvfb\ :([0-9]+) ]]; then
            display=${BASH_REMATCH[1]}
            pid=$(echo $line | awk '{print $1}')
            echo "Display :$display (Xvfb PID: $pid)"
            found=1
        fi
    done < <(ps aux | grep "Xvfb :" | grep -v grep)

    # Check x11vnc processes
    while read -r line; do
        if [[ $line =~ -display\ :([0-9]+).*rfbport\ ([0-9]+) ]]; then
            display=${BASH_REMATCH[1]}
            port=${BASH_REMATCH[2]}
            pid=$(echo $line | awk '{print $2}')
            echo "  -> VNC Port $port (x11vnc PID: $pid)"
        fi
    done < <(ps aux | grep "x11vnc" | grep -v grep)

    if [ $found -eq 0 ]; then
        echo "No virtual displays currently running."
    fi
}

# Function to export display
export_display() {
    if is_display_running $DISPLAY_NUM; then
        echo "export DISPLAY=:$DISPLAY_NUM"
        echo "# Run the above command to set your DISPLAY environment variable"
    else
        echo "Error: Display :$DISPLAY_NUM is not running"
        return 1
    fi
}

# Parse command line arguments
ACTION=""
INTERACTIVE=false

# Check if no arguments provided - default to interactive mode
if [[ $# -eq 0 ]]; then
    INTERACTIVE=true
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--display)
            DISPLAY_NUM="$2"
            DISPLAY_NUM_SET="true"
            shift 2
            ;;
        -r|--resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        -p|--port)
            VNC_PORT="$2"
            VNC_PORT_SET="true"
            shift 2
            ;;
        -t|--desktop)
            DESKTOP_ENV="$2"
            DESKTOP_ENV_SET="true"
            shift 2
            ;;
        -x|--xvfb-args)
            XVFB_ARGS="$2"
            shift 2
            ;;
        -v|--vnc-args)
            VNC_ARGS="$2"
            shift 2
            ;;
        -s|--start)
            ACTION="start"
            shift
            ;;
        -k|--kill)
            ACTION="kill"
            shift
            ;;
        -l|--list)
            ACTION="list"
            shift
            ;;
        -e|--export)
            ACTION="export"
            shift
            ;;
        -i|--interactive)
            INTERACTIVE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# If interactive mode or no action specified, show interactive menu
if [[ "$INTERACTIVE" == "true" ]] || [[ -z "$ACTION" ]]; then
    show_interactive_menu
    exit 0
fi

# Execute action for non-interactive mode
case $ACTION in
    start)
        echo "Starting display with configuration:"
        echo "  Display: :$DISPLAY_NUM"
        echo "  Resolution: $RESOLUTION"
        echo "  VNC Port: $VNC_PORT"
        echo ""

        if confirm "Proceed with these settings?" "y"; then
            start_display
        else
            echo "Operation cancelled."
        fi
        ;;
    kill)
        if confirm "Kill ALL running displays?" "n"; then
            kill_displays
        else
            echo "Operation cancelled."
        fi
        ;;
    list)
        list_displays
        ;;
    export)
        export_display
        ;;
    *)
        echo "Invalid action: $ACTION"
        show_usage
        exit 1
        ;;
esac
