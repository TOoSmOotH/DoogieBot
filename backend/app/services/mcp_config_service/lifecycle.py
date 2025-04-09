# backend/app/services/mcp_config_service/lifecycle.py
"""
Functions for managing the lifecycle (start, stop, restart) of MCP server containers.
"""
import logging
import time # Added import
from typing import Optional, List

from sqlalchemy.orm import Session
from docker.errors import APIError, NotFound as DockerNotFound

from app.schemas.mcp import MCPServerStatus
from .crud import get_config_by_id # Use relative import
from .status import get_config_status # Use relative import
from .docker_utils import _get_docker_client, _get_container_name, _transform_command_to_docker # Use relative import

logger = logging.getLogger(__name__)

def start_server(db: Session, config_id: str) -> Optional[MCPServerStatus]:
    """ Start an MCP server using Docker. """
    db_config = get_config_by_id(db, config_id)
    if not db_config or not db_config.config.get("enabled", False):
        return MCPServerStatus(
            id=config_id, name=db_config.name if db_config else "Unknown",
            enabled=db_config.config.get("enabled", False) if db_config else False,
            status="stopped",
            error_message="Configuration not found or disabled" if not db_config else "Configuration disabled"
        )
    try:
        # Add retry logic for Docker client initialization
        max_retries = 3
        retry_delay = 2
        retry_count = 0
        docker_client = None
        last_exception = None

        while retry_count < max_retries:
            try:
                docker_client = _get_docker_client()
                break  # Successfully got the Docker client, exit the loop
            except Exception as e:
                last_exception = e
                retry_count += 1
                logger.warning(f"Failed to initialize Docker client (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
        
        # If all retries failed, raise an exception
        if docker_client is None:
            error_msg = f"Failed to initialize Docker client after {max_retries} attempts"
            logger.error(error_msg)
            raise Exception(f"{error_msg}: {last_exception}")
        container_name = _get_container_name(config_id)
        container_to_create = True # Flag to indicate if we need to create a new container

        # Check for existing container
        try:
            containers = docker_client.containers.list(all=True, filters={"name": container_name})
            if containers:
                container = containers[0]
                if container.status == "running":
                    logger.info(f"Container '{container_name}' is already running. Skipping start.")
                    container_to_create = False # Don't create if already running
                else:
                    # Container exists but is not running (e.g., exited, created)
                    logger.warning(f"Container '{container_name}' exists but is not running (status: {container.status}). Attempting removal before starting.")
                    try:
                        container.remove(force=True) # Force remove if needed
                        logger.info(f"Successfully removed non-running container '{container_name}'.")
                        # Proceed to create a new container below
                    except APIError as remove_err:
                        # Handle cases where removal is already in progress or container is gone
                        if remove_err.response.status_code == 409 and 'removal in progress' in str(remove_err.explanation).lower():
                            logger.info(f"Container {container_name} already being removed.")
                            container_to_create = False # Assume removal will succeed or another process will handle it
                        elif remove_err.response.status_code == 404:
                            logger.info(f"Container {container_name} already removed (404 on remove attempt).")
                            # Proceed to create
                        else:
                            logger.error(f"Failed to remove non-running container '{container_name}': {remove_err}. Cannot start new one.")
                            raise remove_err # Re-raise error
            else:
                 logger.info(f"Container '{container_name}' not found. Will create.")

        except APIError as list_err:
             logger.error(f"Error checking for existing container '{container_name}': {list_err}")
             raise list_err # Re-raise error

        # Create and start if needed
        if container_to_create:
            logger.info(f"Proceeding to create and start a new container '{container_name}'.")
            # Create a new container
            env_vars = db_config.config.get("env", {}) or {}
            env_vars['DEBUG'] = '1'  # Enable debug logging in MCP container
            command = db_config.config.get("command", "docker")
            args = db_config.config.get("args", [])
            docker_args = _transform_command_to_docker(command, args)

            # --- Start Parsing Logic ---
            image, command_parts, run_options = None, [], []
            options_taking_values = {'-p', '-v', '-e', '--network', '--env-file', '--add-host', '--label', '-l'}
            boolean_flags = {'-i', '-t', '-d', '--rm', '--init', '--privileged'}
            try: run_index = docker_args.index("run"); current_index = run_index + 1
            except ValueError: run_index, current_index = -1, 0

            while current_index < len(docker_args):
                arg = docker_args[current_index]
                if arg.startswith("-"):
                    run_options.append(arg); current_index += 1
                    takes_value = False
                    if arg in options_taking_values: takes_value = True
                    elif arg.startswith("--") and "=" not in arg and arg not in boolean_flags: takes_value = True
                    elif len(arg) == 2 and arg[1] != '-' and arg not in boolean_flags: takes_value = True
                    if takes_value and current_index < len(docker_args) and not docker_args[current_index].startswith("-"):
                        run_options.append(docker_args[current_index]); current_index += 1
                else: image = arg; command_parts = docker_args[current_index + 1:]; break
            if image is None: raise ValueError("Cannot parse Docker image from arguments")
            # --- FIX: Ensure command_to_run is correctly set without -v ---
            command_to_run = command_parts if command_parts else None
            # --- End FIX ---
            # --- End Parsing Logic ---

            # --- Map run_options to docker-py kwargs ---
            run_kwargs = {}
            i = 0
            while i < len(run_options):
                opt = run_options[i]
                if opt == '-i': run_kwargs['stdin_open'] = True; i += 1
                elif opt == '-t': run_kwargs['tty'] = True; i += 1
                elif opt == '-d': run_kwargs['detach'] = True; i += 1
                # --- FIX: Restore --rm mapping ---
                elif opt == '--rm': run_kwargs['auto_remove'] = True; i += 1
                # --- End FIX ---
                elif opt == '--init': run_kwargs['init'] = True; i += 1
                elif opt == '--privileged': run_kwargs['privileged'] = True; i += 1
                elif opt == '-p' and i + 1 < len(run_options):
                    ports_dict = run_kwargs.get('ports', {}); parts = run_options[i+1].split(':')
                    if len(parts) == 2: ports_dict[f'{parts[1]}/tcp'] = parts[0]
                    elif len(parts) == 3: ports_dict[f'{parts[1]}/{parts[2]}'] = parts[0]
                    run_kwargs['ports'] = ports_dict; i += 2
                elif opt == '-v' and i + 1 < len(run_options):
                    volumes_list = run_kwargs.get('volumes', []); volumes_list.append(run_options[i+1])
                    run_kwargs['volumes'] = volumes_list; i += 2
                elif opt == '-e' and i + 1 < len(run_options): i += 2 # Handled below
                elif (opt == '--label' or opt == '-l') and i + 1 < len(run_options):
                     labels_dict = run_kwargs.get('labels', {})
                     if '=' in run_options[i+1]: key, value = run_options[i+1].split('=', 1); labels_dict[key] = value
                     else: labels_dict[run_options[i+1]] = ""
                     run_kwargs['labels'] = labels_dict; i += 2
                elif opt.startswith("--network") and i + 1 < len(run_options): run_kwargs['network'] = run_options[i+1]; i += 2
                elif opt.startswith("--network=") : run_kwargs['network'] = opt.split("=", 1)[1]; i += 1
                else:
                     logger.warning(f"Ignoring unknown/unhandled Docker option during mapping: {opt}")
                     if i + 1 < len(run_options) and not run_options[i+1].startswith("-"):
                         if not (opt.startswith("--") and "=" in opt):
                             is_bool_flag = opt in boolean_flags or (opt.startswith("--") and opt not in options_taking_values and "=" not in opt)
                             if not is_bool_flag: i += 1
                     i += 1
            # --- End Mapping ---

            final_env = db_config.config.get("env", {}) or {}
            final_env['DEBUG'] = '1' # Add DEBUG env var
            i = 0
            while i < len(run_options):
                if run_options[i] == '-e' and i + 1 < len(run_options):
                    env_item = run_options[i+1]
                    if '=' in env_item: key, value = env_item.split('=', 1); final_env[key] = value
                    i += 2
                else: i += 1

            run_kwargs['detach'] = True
            run_kwargs['auto_remove'] = False # Explicitly disable auto-remove for debugging/restarts
            run_kwargs['name'] = container_name
            run_kwargs['restart_policy'] = {"Name": "always"} # Ensure container restarts

            # --- Explicitly set stream attach options ---
            # Ensure stdin is open and attachable (usually handled by -i mapping)
            if 'stdin_open' not in run_kwargs:
                run_kwargs['stdin_open'] = True
            # Explicitly enable attaching stdout and stderr for reading responses/errors
            run_kwargs['stdout'] = True
            run_kwargs['stderr'] = True
            # Ensure TTY is false for stream demultiplexing
            run_kwargs['tty'] = False
            # --- End stream attach options ---

            # --- Check for existing container AGAIN before run --- (Final idempotency check)
            try:
                existing_container = docker_client.containers.get(container_name)
                logger.warning(f"Container '{container_name}' unexpectedly exists before run attempt. Status: {existing_container.status}")
                if existing_container.status == "running":
                    return get_config_status(db, config_id) # Already running, return status
                else:
                    # Should have been removed earlier, but try again just in case
                    try:
                        existing_container.remove(force=True)
                        logger.info(f"Removed existing non-running container '{container_name}' before final run attempt.")
                    except APIError as final_remove_err:
                         logger.error(f"Failed final removal attempt for '{container_name}': {final_remove_err}")
                         raise final_remove_err
            except DockerNotFound:
                pass # Good, container doesn't exist
            except APIError as final_get_err:
                logger.error(f"Error checking container before final run attempt: {final_get_err}")
                raise final_get_err
            # --- End Check ---

            logger.debug(f"Running container '{container_name}' with image='{image}', command={command_to_run}, env={final_env}, kwargs={run_kwargs}")
            try:
                container = docker_client.containers.run(image=image, command=command_to_run, environment=final_env, **run_kwargs)
                logger.info(f"Created and started new MCP container: {container_name}")
            except APIError as e:
                 # Catch potential 409 conflict from run if check somehow failed, or other API errors
                 if e.response.status_code == 409:
                     logger.warning(f"Container '{container_name}' already exists (caught during run attempt). Status: {get_config_status(db, config_id).status}")
                 else:
                     logger.error(f"Docker API error during run: {e}"); raise

        # Return status after attempting start/create
        return get_config_status(db, config_id)

    except Exception as e:
        logger.exception(f"Error starting MCP server: {e}") # Use exception for full traceback
        config_name = db_config.name if db_config else "Unknown";
        config_enabled = db_config.config.get("enabled", False) if db_config else False
        return MCPServerStatus(id=config_id, name=config_name, enabled=config_enabled, status="error", error_message=str(e))

# --- Corrected stop_server function ---
def stop_server(db: Session, config_id: str) -> Optional[MCPServerStatus]:
    """ Stop an MCP server Docker container. """
    db_config = get_config_by_id(db, config_id)
    if not db_config: return None
    try:
        docker_client = _get_docker_client()
        container_name = _get_container_name(config_id)
        try:
            containers = docker_client.containers.list(all=True, filters={"name": container_name})
            if containers:
                container = containers[0]
                if container.status == "running":
                    try:
                        container.stop(timeout=10)
                        logger.info(f"Stopped MCP container: {container_name}")
                    except APIError as stop_error:
                        logger.error(f"Failed to stop container {container_name}: {stop_error}")
                        raise stop_error # Re-raise if stop fails critically

                # --- FIX: Only remove if AutoRemove is not set ---
                if not container.attrs.get('HostConfig', {}).get('AutoRemove', False):
                    try:
                        container.remove()
                        logger.info(f"Removed stopped MCP container (AutoRemove=False): {container_name}")
                    except APIError as remove_error:
                        # Handle cases where removal is already in progress or container is gone
                        if remove_error.response.status_code == 409 and 'removal in progress' in str(remove_err.explanation).lower():
                            logger.info(f"Container {container_name} already being removed.")
                        elif remove_error.response.status_code == 404:
                            logger.info(f"Container {container_name} already removed (404 on remove attempt).")
                        else:
                            logger.error(f"Failed to remove container {container_name}: {remove_error}")
                            # Don't raise here for remove error, just log it
                else:
                    logger.info(f"Container {container_name} has AutoRemove=True, skipping explicit removal.")
                # --- End FIX ---

        except DockerNotFound:
            logger.info(f"Container {container_name} not found, nothing to stop/remove.")
        except APIError as e:
            logger.error(f"Docker API error during stop/remove: {e}")
            raise # Re-raise other API errors

        # Return the status after attempting stop/remove
        return get_config_status(db, config_id)

    except Exception as e:
        logger.exception(f"Error stopping MCP server: {e}") # Use exception
        return MCPServerStatus(id=config_id, name=db_config.name, enabled=db_config.config.get("enabled", False), status="error", error_message=str(e))
# --- End corrected stop_server function ---

def restart_server(db: Session, config_id: str) -> Optional[MCPServerStatus]:
    """ Restart an MCP server Docker container. """
    logger.info(f"Attempting to restart server {config_id}")
    stop_result = stop_server(db, config_id)
    # Add a small delay to allow Docker daemon to fully process removal if needed
    time.sleep(1)
    current_status = get_config_status(db, config_id)
    if current_status and current_status.status == "running":
         logger.warning(f"Server {config_id} did not stop cleanly before restart attempt, current status: {current_status.status}. Attempting start anyway.")
    elif current_status and current_status.status != "stopped":
         logger.warning(f"Server {config_id} in unexpected state '{current_status.status}' before restart attempt. Attempting start anyway.")
    else:
        logger.info(f"Server {config_id} stopped cleanly (or was not found). Proceeding with start.")

    return start_server(db, config_id)