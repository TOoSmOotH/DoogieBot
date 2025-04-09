"""
XML parser utilities for Anthropic tool handling.

These utilities help extract structured data from Anthropic's XML-based tool use format.
"""

import re
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def extract_tool_call_from_xml(xml_text: str) -> Optional[Dict[str, Any]]:
    """
    Extract a tool call from Anthropic's XML format.
    
    Args:
        xml_text: XML string containing an Anthropic tool_use block
        
    Returns:
        Dictionary with tool name and arguments, or None if parsing fails
    """
    if not xml_text or "<tool_use>" not in xml_text:
        logger.warning("No tool_use block found in XML text")
        return None
    
    try:
        # Extract the tool_use block
        tool_use_match = re.search(r'<tool_use>(.*?)</tool_use>', xml_text, re.DOTALL)
        if not tool_use_match:
            logger.warning("Failed to extract tool_use block from XML")
            return None
            
        tool_use_block = tool_use_match.group(1).strip()
        
        # Extract tool name
        name_match = re.search(r'<tool_name>(.*?)</tool_name>', tool_use_block, re.DOTALL)
        if not name_match:
            logger.warning("No tool_name found in tool_use block")
            return None
            
        tool_name = name_match.group(1).strip()
        
        # Extract parameters block
        params_match = re.search(r'<parameters>(.*?)</parameters>', tool_use_block, re.DOTALL)
        params_block = params_match.group(1).strip() if params_match else ""
        
        # Parse individual parameters
        params = {}
        param_matches = re.findall(r'<([^>]+)>(.*?)</\1>', params_block, re.DOTALL)
        
        for param_name, param_value in param_matches:
            # Try to convert values to appropriate types
            try:
                # Try to parse as number, boolean, etc.
                parsed_value = json.loads(param_value.strip())
                params[param_name] = parsed_value
            except json.JSONDecodeError:
                # Keep as string if not valid JSON
                params[param_name] = param_value.strip()
        
        # Return a properly formatted tool call
        return {
            "name": tool_name,
            "arguments": json.dumps(params)
        }
        
    except Exception as e:
        logger.error(f"Error parsing tool use XML: {e}")
        return None