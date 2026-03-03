"""Main entry point for the Eatventure bot."""

import logging
import sys
import time
from pynput import keyboard

from bot import EatventureBot
from core.logger import setup_logger

logger = setup_logger("main")

bot_instance = None
should_exit = False

def on_press(key):
    global bot_instance, should_exit
    try:
        if key == keyboard.Key.esc:
            logger.critical("Emergency stop triggered via ESC key!")
            if bot_instance:
                bot_instance.stop()
            should_exit = True
            return False # Stop listener
            
        if hasattr(key, 'char') and key.char:
            if key.char == 'z':
                if bot_instance:
                    if not bot_instance.running:
                        bot_instance.start()
                    else:
                        bot_instance.stop()
            elif key.char == 'p':
                logger.info("Exiting program via P key...")
                should_exit = True
    except Exception as e:
        logger.error(f"Error in keyboard listener: {e}")

def main():
    global bot_instance, should_exit
    
    logger.info("=" * 60)
    logger.info("Eatventure Bot (Refactored Architecture)")
    logger.info("=" * 60)
    
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    try:
        bot = EatventureBot()
        bot_instance = bot
        
        logger.info("Bot initialized and ready.")
        logger.info("Press Z to START/STOP the bot.")
        logger.info("Press P to EXIT the program.")
        
        while not should_exit:
            if bot.running:
                bot.step()
            time.sleep(0.015) # Aligned with 60FPS frame timing
        
        logger.info("Program exiting...")
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (Ctrl+C).")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        listener.stop()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
