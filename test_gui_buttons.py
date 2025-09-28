"""
Test script to verify GUI buttons are created properly
"""

import tkinter as tk
from classroom_analyzer_gui import ClassroomAnalyzerGUI

def test_gui_buttons():
    """Test if all GUI buttons are created"""
    print("🧪 Testing GUI Button Creation")
    print("=" * 40)
    
    # Create a test window
    root = tk.Tk()
    root.title("Test GUI Buttons")
    root.geometry("800x600")
    
    try:
        # Create the GUI app
        app = ClassroomAnalyzerGUI(root)
        
        # Check if automated processing buttons exist
        buttons_to_check = [
            'auto_process_button',
            'stop_auto_button', 
            'history_button',
            'folder_button'
        ]
        
        print("🔍 Checking for automated processing buttons...")
        for button_name in buttons_to_check:
            if hasattr(app, button_name):
                button = getattr(app, button_name)
                if button:
                    print(f"✅ {button_name}: Found")
                    print(f"   Text: {button.cget('text')}")
                    print(f"   State: {button.cget('state')}")
                else:
                    print(f"❌ {button_name}: Not created")
            else:
                print(f"❌ {button_name}: Attribute not found")
        
        # Check if the buttons are visible
        print("\n🔍 Checking button visibility...")
        for button_name in buttons_to_check:
            if hasattr(app, button_name):
                button = getattr(app, button_name)
                if button:
                    try:
                        # Check if button is packed
                        info = button.pack_info()
                        if info:
                            print(f"✅ {button_name}: Packed and visible")
                        else:
                            print(f"⚠️ {button_name}: Not packed")
                    except:
                        print(f"❌ {button_name}: Packing error")
        
        print("\n🎉 GUI button test completed!")
        
    except Exception as e:
        print(f"❌ Error testing GUI buttons: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        root.destroy()

if __name__ == "__main__":
    test_gui_buttons()