#!/bin/bash

#==============================================================================
# Python script
#==============================================================================
# Description: Interactive menu system for managing the AI HelpDesk backend
# Author: Smart-Doc Team
# Version: 2.0
# Last Modified: $(date +%Y-%m-%d)
#
# Features:
# - Server management (Granian, Streamlit, Ollama)
# - Database model conversion (UUID7)
# - Package management
# - Virtual environment operations
# - File cleanup utilities
#
# Requirements:
# - bash 4.0+
# - fzf (optional, for enhanced menu)
# - Python 3.8+
# - Node.js (for nodemon)
#==============================================================================

#==============================================================================
# GLOBAL CONSTANTS AND CONFIGURATION
#==============================================================================

# Terminal color definitions
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly DIM='\033[2m'
readonly NC='\033[0m'  # No Color

# File paths
readonly TARGET_FILE="src/Domain/base_entities.py"
readonly PACKAGES_FILE="packages.txt"
readonly UPLOADS_DIR="static/"

# Spinner animation characters
readonly SPINNER_CHARS=('' '' '' '' '' '' '' '' '')

# Application settings
readonly SERVER_PORT=8080
readonly STREAMLIT_PORT=1337

#==============================================================================
# ENVIRONMENT SETUP
#==============================================================================

# Initialize color support for non-interactive terminals
setup_colors() {
    if [ -z "$TERM" ] || [ "$TERM" = "dumb" ]; then
        # Disable colors for dumb terminals
        RED='' GREEN='' YELLOW='' BLUE='' CYAN='' BOLD='' DIM='' NC=''
    fi
}

#==============================================================================
# UI/UX FUNCTIONS
#==============================================================================

# Spinner animation variables
SPINNER_PID=""

# Start loading spinner with custom message
start_spinner() {
    local message="${1:-Loading...}"
    printf "${CYAN}%s${NC}" "$message"
    
    (
        while true; do
            for char in "${SPINNER_CHARS[@]}"; do
                printf "\r${CYAN}%s %s${NC}" "$char" "$message"
                sleep 0.1
            done
        done
    ) &
    SPINNER_PID=$!
}

# Stop running spinner and clean up
stop_spinner() {
    if [ -n "$SPINNER_PID" ]; then
        kill $SPINNER_PID 2>/dev/null
        wait $SPINNER_PID 2>/dev/null
        SPINNER_PID=""
    fi
    printf "\r\033[K"  # Clear line
}

# Print formatted success message
print_success() {
    echo -e "${GREEN}✅ ${NC} $1"
}

# Print formatted error message
print_error() {
    echo -e "${RED}❌ ${NC} $1"
}

# Print formatted warning message
print_warning() {
    echo -e "${YELLOW}⚠️ ${NC} $1"
}

# Print formatted info message
print_info() {
    echo -e "${BLUE}ℹ️️ ${NC} $1"
}

#==============================================================================
# MENU SYSTEM
#==============================================================================

# Check if fzf is available for enhanced menu
has_fzf() {
    command -v fzf &>/dev/null
}

# Display enhanced menu using fzf
menu_fzf() {
    local height=${1:-15}
    echo "$MENU_OPTIONS_JSON" | fzf \
        --height=$height \
        --border \
        --layout=reverse \
        --prompt="AI HelpDesk > " \
        --color=fg:#e5e9f0,bg:#1e1e2e,preview-bg:#181825,hl:#cba6f7,fg+:#cdd6f4,bg+:#313244,border:#585b70,spinner:#f5c2e7,header:#f5c2e7
}

# Display native terminal menu (fallback)
menu_native() {
    clear
    echo -e "${BOLD}${CYAN}=================================================${NC}"
    echo -e "${BOLD}${CYAN}|${NC}     ${BOLD}AI HelpDesk - Backend Manager${NC}     ${CYAN}|${NC}"
    echo -e "${BOLD}${CYAN}=================================================${NC}"
    echo ""
    
    # Display menu options with selection indicator
    for i in "${!MENU_LABELS[@]}"; do
        if [ "$i" -eq "$SELECTED" ]; then
            echo -e "  ${GREEN}>${NC} ${BOLD}$((i+1))) ${MENU_LABELS[$i]}${NC}"
        else
            echo -e "    ${DIM}$((i+1))) ${MENU_LABELS[$i]}${NC}"
        fi
    done
    
    echo ""
    echo -e "${DIM} Move | Enter select | q quit${NC}"
}

#==============================================================================
# SERVER MANAGEMENT FUNCTIONS
#==============================================================================

# Start the main application server using Granian
run_server() {
    print_info "Starting application server..."
    dev
}

# Development server with auto-reload using nodemon
dev() {
    print_info "Starting development server with auto-reload..."
    npx nodemon --watch . --ext yaml --exec "set PYTHONPATH=%cd%\\src && granian --interface asgi --port $SERVER_PORT src.main:app"
}

# Start Streamlit UI server
run_streamlit() {
    print_info "Starting Streamlit UI server..."
    set PYTHONPATH=%cd%\src && \
    streamlit run ui/app.py \
        --server.port $STREAMLIT_PORT \
        --server.runOnSave true \
        --server.headless true
}

# Start Ollama AI service
run_ollama() {
    print_info "Starting Ollama AI service..."
    ollama serve
}

# Fetch OpenAPI spec from server to JSON file
fetch_openapi() {
    print_info "Fetching OpenAPI spec from localhost:$SERVER_PORT..."
    curl -s http://localhost:$SERVER_PORT/openapi.json -o openapi.json
    if [ -f "openapi.json" ]; then
        print_success "OpenAPI spec saved to openapi.json"
        python -c "import json; f=open('openapi.json','r',encoding='utf-8'); d=json.load(f); json.dump(d,open('openapi.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)"
        print_success "OpenAPI JSON formatted"
    else
        print_error "Failed to fetch OpenAPI spec"
    fi
}

# Generate OpenAPI spec to YAML file
gen_openapi() {
    print_info "Generating OpenAPI spec to openapi.yaml..."
    curl -s http://localhost:$SERVER_PORT/openapi.json | python -c "
import json, sys, yaml
data = json.load(sys.stdin)
print(yaml.dump(data, allow_unicode=True, sort_keys=False))
" > openapi.yaml
    if [ -f "openapi.yaml" ]; then
        print_success "OpenAPI spec saved to openapi.yaml"
    else
        print_error "Failed to generate OpenAPI spec"
    fi
}

# Start Windows Subsystem for Linux
run_wsl() {
    print_info "Starting WSL..."
    wsl
}

#==============================================================================
# DATABASE OPERATIONS
#==============================================================================

# Convert database model IDs to UUID7 format
convert_uuid() {
    print_info "Converting database IDs to UUID7 format..."
    
    # Prompt for database name
    echo -n "Nhâp tên database: "
    read -e db_name
    
    # Validate database name
    if [ -z "$db_name" ]; then
        print_error "Tên database không thê rông!"
        return 1
    fi
    
    # Generate SQLModel code from database
    if ! python -m sqlacodegen mysql://root:@localhost:3306/$db_name \
        --generator sqlmodels --outfile "$TARGET_FILE" 2>/dev/null; then
        print_error "Failed to generate SQLModel code from database"
        return 1
    fi
    
    # Verify target file exists
    if [ ! -f "$TARGET_FILE" ]; then
        print_error "Target file not found: $TARGET_FILE"
        return 1
    fi

    # Add uuid6 import if not present
    if ! grep -q "import uuid6" "$TARGET_FILE"; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' '1i\'$'\n''import uuid6' "$TARGET_FILE"
        else
            sed -i '1iimport uuid6' "$TARGET_FILE"
        fi
        print_success "Added 'import uuid6' to target file"
    fi

    # Replace UUID generation pattern
    local search_pattern="sa_column=Column('id', CHAR(36), primary_key=True, server_default=text('(uuid())'))"
    local replace_pattern="default_factory=lambda: str(uuid6.uuid7()), sa_column=Column(CHAR(36), primary_key=True)"

    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s|$search_pattern|$replace_pattern|g" "$TARGET_FILE"
    else
        sed -i "s|$search_pattern|$replace_pattern|g" "$TARGET_FILE"
    fi
    
    print_success "Database model conversion completed!"
}

# Convert DBML to MySQL SQL
convert_dbml() {
    print_info "Convert DBML to MySQL SQL..."
    
    # Enable tab completion
    bind '"\t": menu-complete'
    
    # Prompt for database name
    echo -n "Nhâp tên database: "
    read -e db_name
    
    # Validate database name
    if [ -z "$db_name" ]; then
        print_error "Tên database không thê rông!"
        return 1
    fi
    
    # Convert DBML to MySQL SQL
    print_info "Dang convert $db_name sang MySQL SQL"
    dbml2sql "$db_name" --mysql -o "${db_name}.sql"
    
    # Check conversion result
    if [ $? -eq 0 ]; then
        print_success "Convert hoàn tât! File output: ${db_name}.sql"
        
        # Fix PostgreSQL syntax to MySQL
        print_info "Fixing syntax for MySQL compatibility..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            sed -i '' 's/DEFAULT (now())/DEFAULT CURRENT_TIMESTAMP/g' "${db_name}.sql"
            sed -i '' 's/-- Database: PostgreSQL/-- Database: MySQL/g' "${db_name}.sql"
        else
            # Linux/WSL
            sed -i 's/DEFAULT (now())/DEFAULT CURRENT_TIMESTAMP/g' "${db_name}.sql"
            sed -i 's/-- Database: PostgreSQL/-- Database: MySQL/g' "${db_name}.sql"
        fi
        
        print_success "MySQL syntax fixes applied!"
    else
        print_error "Convert thât bai!"
        return 1
    fi
}

#==============================================================================
# FILE MANAGEMENT UTILITIES
#==============================================================================

# Clean up upload directories with UUID-named folders
clear_uploads() {
    print_info "Cleaning up UUID-named upload directories..."
    
    if [ ! -d "$UPLOADS_DIR" ]; then
        print_warning "Uploads directory not found: $UPLOADS_DIR"
        return 1
    fi
    
    local deleted_count
    deleted_count=$(find "$UPLOADS_DIR" -type d -name "*-*-*-*-*" -exec rm -rf {} + 2>/dev/null | wc -l)
    print_success "Cleaned up $deleted_count UUID directories"
}

#==============================================================================
# PACKAGE MANAGEMENT
#==============================================================================

# List packages that are not required dependencies
list_packages() {
    echo -e "\n${BOLD}Non-required packages:${NC}"
    pip list --not-required 2>/dev/null | grep -v "^pip " || \
        print_info "No non-required packages found"
}

# Install packages from requirements file
install_packages() {
    if [ ! -f "$PACKAGES_FILE" ]; then
        print_error "Packages file not found: $PACKAGES_FILE"
        return 1
    fi
    
    print_info "Installing packages from $PACKAGES_FILE..."
    if pip install -r "$PACKAGES_FILE" 2>/dev/null; then
        print_success "Package installation completed"
    else
        print_error "Package installation failed"
        return 1
    fi
}

# Uninstall all currently installed packages
uninstall_all_packages() {
    print_warning "Uninstalling all packages..."
    
    local temp_file="temp_packages_to_delete.txt"
    pip freeze > "$temp_file" 2>/dev/null
    
    if [ -s "$temp_file" ]; then
        pip uninstall -r "$temp_file" -y 2>/dev/null
        print_success "All packages uninstalled"
    else
        print_info "No packages to uninstall"
    fi
    
    rm -f "$temp_file"
}

#==============================================================================
# VIRTUAL ENVIRONMENT MANAGEMENT
#==============================================================================

# Create Python virtual environment
install_venv() {
    if [ -d "venv" ] || [ -d ".venv" ]; then
        print_warning "Virtual environment already exists"
        return 0
    fi
    
    print_info "Creating Python virtual environment..."
    if python -m venv .venv 2>/dev/null; then
        print_success "Virtual environment created successfully"
        print_info "Run 'Activate venv' from the menu to use it"
    else
        print_error "Failed to create virtual environment"
        return 1
    fi
}

# Activate virtual environment (cross-platform)
activate_venv() {
    local venv_paths=(
        ".venv/bin/activate"
        "venv/bin/activate"
        ".venv/Scripts/activate"
        "venv/Scripts/activate"
    )
    
    for path in "${venv_paths[@]}"; do
        if [ -f "$path" ]; then
            source "$path"
            local platform="Unknown"
            [[ "$path" == *"Scripts"* ]] && platform="Windows" || platform="Linux/WSL"
            print_success "Activated virtual environment ($platform)"
            return 0
        fi
    done
    
    print_error "No virtual environment found"
    print_info "Use 'Install venv' option first"
    return 1
}

# Deactivate virtual environment
deactivate_venv() {
    if command -v deactivate &>/dev/null; then
        deactivate
        print_success "Virtual environment deactivated"
    else
        print_info "No active virtual environment"
    fi
}

#==============================================================================
# MENU CONFIGURATION
#==============================================================================

# Menu function mappings
readonly MENU_OPTIONS=(
    "run_server"
    "run_streamlit"
    "run_ollama"
    "run_wsl"
    "fetch_openapi"
    "gen_openapi"
    "convert_uuid"
    "convert_dbml"
    "clear_uploads"
    "list_packages"
    "install_packages"
    "uninstall_all_packages"
    "install_venv"
    "activate_venv"
    "deactivate_venv"
)

# Menu display labels
readonly MENU_LABELS=(
    "Server (Granian)"
    "Streamlit"
    "Ollama Serve"
    "WSL"
    "Fetch OpenAPI JSON"
    "Gen OpenAPI YAML"
    "Convert Model (UUID7)"
    "Convert DBML to SQL"
    "Clear uploads"
    "List packages"
    "Install packages"
    "Uninstall packages"
    "Install venv"
    "Activate venv"
    "Deactivate venv"
)

# Current menu selection index
SELECTED=0

#==============================================================================
# MAIN MENU HANDLER
#==============================================================================

# Main menu loop with keyboard navigation
main_menu() {
    if has_fzf; then
        # Enhanced menu using fzf
        echo -e "${BOLD}${CYAN}Starting with fzf menu...${NC}"
        sleep 0.5
        selected=$(printf '%s\n' "${MENU_LABELS[@]}" | fzf \
            --height=20 \
            --border \
            --layout=reverse \
            --prompt="AI HelpDesk > " \
            --color=bg:#1e1e2e,fg:#cdd6f4,hl:#cba6f7,pointer:#f5c2e7,header:#89b4fa \
            --preview="echo 'Select an action to execute'")
        
        # Execute selected function
        for i in "${!MENU_LABELS[@]}"; do
            if [ "${MENU_LABELS[$i]}" = "$selected" ]; then
                clear
                ${MENU_OPTIONS[$i]}
                return
            fi
        done
    else
        # Native terminal menu with keyboard navigation
        while true; do
            menu_native
            
            # Read single key input
            read -n 1 key
            
            # Handle arrow keys
            if [ "$key" = $'\e' ]; then
                read -n 1 -t 0.1 seq
                if [ "$seq" = "[" ]; then
                    read -n 1 -t 0.1 dir
                    case $dir in
                        A)  # Up arrow
                            ((SELECTED--))
                            if [ "$SELECTED" -lt 0 ]; then 
                                SELECTED=$((${#MENU_LABELS[@]}-1))
                            fi
                            ;;
                        B)  # Down arrow
                            ((SELECTED++))
                            if [ "$SELECTED" -ge ${#MENU_LABELS[@]} ]; then 
                                SELECTED=0
                            fi
                            ;;
                    esac
                fi
            # Handle Enter key
            elif [ "$key" = "" ]; then
                clear
                ${MENU_OPTIONS[$SELECTED]}
                return
            # Handle quit key
            elif [ "$key" = "q" ] || [ "$key" = "Q" ]; then
                clear
                echo -e "${CYAN}Goodbye!${NC}"
                exit 0
            # Handle number keys (1-9)
            elif [[ "$key" =~ [1-9] ]]; then
                idx=$((key-1))
                if [ "$idx" -lt ${#MENU_LABELS[@]} ]; then
                    clear
                    ${MENU_OPTIONS[$idx]}
                    return
                fi
            fi
        done
    fi
}

#==============================================================================
# SCRIPT EXECUTION
#==============================================================================

# Initialize environment and start main menu
main() {
    # Setup color support
    setup_colors
    
    # Display welcome message
    clear
    echo -e "${BOLD}${CYAN}AI HelpDesk Backend Manager${NC}"
    echo -e "${DIM}Version 2.0 - Enhanced Management Script${NC}"
    echo ""
    
    # Start main menu
    main_menu
}

# Execute main function
main
