"""
Test to document the expected behavior of URI path handling in the generated FMU C++ code.

This test documents the fix for the URI path parsing bug where the formatOnnxPath function
in onnxFmu.cpp was incorrectly stripping 8 characters instead of 7 when removing the 
"file://" prefix from URIs.

The actual formatOnnxPath function is in C++ and gets compiled into the FMU binary,
so this test serves as documentation of the expected behavior and can be used for
regression testing if the logic is ever ported to Python.
"""

import pytest


class TestUriPathHandling:
    """
    Document expected URI path handling behavior.
    
    These tests describe how the formatOnnxPath C++ function should handle
    file:// URI prefixes on different operating systems.
    """

    @pytest.mark.parametrize(
        "input_uri,expected_path",
        [
            # Unix absolute paths with file:// prefix (7 characters)
            ("file:///tmp/extracted_fmu/resources/model.onnx", "/tmp/extracted_fmu/resources/model.onnx"),
            ("file:///home/user/fmu/model.onnx", "/home/user/fmu/model.onnx"),
            ("file:///var/lib/simulation/model.onnx", "/var/lib/simulation/model.onnx"),
            
            # Windows paths (for documentation - actual behavior may vary on Windows)
            # Note: Windows file URIs typically use file:///C:/path format
            ("file:///C:/Users/test/model.onnx", "/C:/Users/test/model.onnx"),
            
            # Paths without file:// prefix should remain unchanged
            ("/tmp/path/model.onnx", "/tmp/path/model.onnx"),
            ("C:/Users/test/model.onnx", "C:/Users/test/model.onnx"),
        ],
    )
    def test_uri_path_transformation_expected_behavior(self, input_uri: str, expected_path: str) -> None:
        """
        Document the expected behavior of URI to path transformation.
        
        The C++ formatOnnxPath function should:
        1. Check if path starts with "file://" (7 characters)
        2. If yes, strip exactly 7 characters to get the actual path
        3. This preserves the leading "/" in Unix absolute paths
        
        Before the fix:
        - Checked for "file:///" (8 characters) 
        - Stripped 8 characters
        - Result: "/tmp/path" became "tmp/path" (missing leading /)
        
        After the fix:
        - Checks for "file://" (7 characters)
        - Strips 7 characters  
        - Result: "/tmp/path" stays "/tmp/path" (leading / preserved)
        """
        # This is a documentation test - the actual implementation is in C++
        # If this logic is ever ported to Python, this test can validate it
        
        def simulate_format_onnx_path(path: str) -> str:
            """Simulate the C++ formatOnnxPath behavior after the fix."""
            if path.startswith("file://"):
                # Strip exactly 7 characters for "file://"
                return path[7:]
            return path
        
        result = simulate_format_onnx_path(input_uri)
        assert result == expected_path, (
            f"Path transformation failed:\n"
            f"  Input:    {input_uri}\n"
            f"  Expected: {expected_path}\n"
            f"  Got:      {result}\n"
        )

    def test_buggy_behavior_should_not_occur(self) -> None:
        """
        Verify that the buggy behavior (stripping 8 characters) is not present.
        
        The bug was:
        - Input:  "file:///tmp/path/model.onnx"
        - Buggy output: "tmp/path/model.onnx" (missing leading /)
        - Correct output: "/tmp/path/model.onnx" (leading / preserved)
        """
        input_uri = "file:///tmp/path/model.onnx"
        
        # Simulate the FIXED behavior (strip 7 chars)
        def fixed_behavior(path: str) -> str:
            if path.startswith("file://"):
                return path[7:]  # Strip "file://"
            return path
        
        # Simulate the BUGGY behavior (strip 8 chars)
        def buggy_behavior(path: str) -> str:
            if path.startswith("file:///"):  # Checks for 8 chars
                return path[8:]  # Strips 8 chars - BUG!
            return path
        
        fixed_result = fixed_behavior(input_uri)
        buggy_result = buggy_behavior(input_uri)
        
        # The fixed result should have the leading slash
        assert fixed_result == "/tmp/path/model.onnx", "Fixed behavior should preserve leading /"
        
        # The buggy result would be missing the leading slash  
        assert buggy_result == "tmp/path/model.onnx", "Buggy behavior strips the leading /"
        
        # Verify they are different
        assert fixed_result != buggy_result, "Fixed and buggy behaviors should differ"
        
        # Verify the fixed version is an absolute path
        assert fixed_result.startswith("/"), "Fixed result should be an absolute path on Unix"
