import io
import logging
import os

import discord
import discord.app_commands
import requests
from blockchain_utils import list_anchors
from discord.ext import commands
from dotenv import load_dotenv
from encryption_utils import decrypt_data, encrypt_data, generate_key
from gtts import gTTS
from PIL import ImageGrab

from app import wdbx
from plugins.web_scraper.utils import scrape_url

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Discord bot
bot = commands.Bot(command_prefix="!")

enabled_intents = discord.Intents.all()


# Sync slash commands on ready
def setup_slash_commands():
    @bot.event
    async def on_ready():
        await bot.tree.sync()
        logging.info(f"Discord Bot ready: {bot.user} (ID: {bot.user.id})")

    @bot.tree.command(
        name="scrape",
        description="Scrape a URL and return page title and first paragraph",
    )
    async def slash_scrape(interaction: discord.Interaction, url: str):
        await interaction.response.defer(thinking=True)
        try:
            data = scrape_url(url)
            embed = discord.Embed(
                title=data.get("title", ""), description=data.get("first_paragraph", "")
            )
            await interaction.followup.send(embed=embed)
        except Exception as e:
            await interaction.followup.send(f"Error scraping URL: {e}")

    @bot.tree.command(name="anchors", description="Show recent blockchain anchors")
    async def slash_anchors(interaction: discord.Interaction):
        try:
            entries = list_anchors(limit=10)
            lines = [f"{e.get('file_path')}: {e.get('file_hash')}" for e in entries]
            content = "\n".join(lines) or "No anchors found."
            await interaction.response.send_message(f"**Anchored Hashes:**\n{content}")
        except Exception as e:
            await interaction.response.send_message(f"Error fetching anchors: {e}")

    @bot.tree.command(
        name="genkey", description="Generate an AES-GCM key and DM it to you"
    )
    async def slash_genkey(interaction: discord.Interaction):
        key = generate_key()
        await interaction.user.send(f"🔑 Your AES-GCM key: `{key.hex()}`")
        await interaction.response.send_message(
            f"🔒 Key generated and sent via DM, {interaction.user.mention}."
        )

    @bot.tree.command(name="encrypt", description="Encrypt text with AES-GCM key")
    async def slash_encrypt(
        interaction: discord.Interaction, key_hex: str, plaintext: str
    ):
        try:
            key = bytes.fromhex(key_hex)
            result = encrypt_data(key, plaintext.encode("utf-8"))
            await interaction.user.send(
                f"🔐 Ciphertext: `{result['ciphertext']}`, nonce: `{result['nonce']}`"
            )
            await interaction.response.send_message(
                f"✅ Encryption complete. Check your DMs, {interaction.user.mention}."
            )
        except Exception as e:
            await interaction.response.send_message(f"Encryption error: {e}")

    @bot.tree.command(name="decrypt", description="Decrypt AES-GCM ciphertext")
    async def slash_decrypt(
        interaction: discord.Interaction,
        key_hex: str,
        nonce_hex: str,
        ciphertext_hex: str,
    ):
        try:
            key = bytes.fromhex(key_hex)
            plaintext = decrypt_data(key, nonce_hex, ciphertext_hex)
            await interaction.response.send_message(
                f"🔓 Decrypted text: {plaintext.decode('utf-8')}"
            )
        except Exception as e:
            await interaction.response.send_message(f"Decryption error: {e}")

    @bot.tree.command(name="ping", description="Ping the WDBX service")
    async def slash_ping(interaction: discord.Interaction):
        try:
            healthy = wdbx.ping()
            status = "✅ Healthy" if healthy else "❌ Unresponsive"
            await interaction.response.send_message(f"WDBX ping status: {status}")
        except Exception as e:
            await interaction.response.send_message(f"Ping error: {e}")

    @bot.tree.command(name="shards", description="Show health of WDBX shards")
    async def slash_shards(interaction: discord.Interaction):
        try:
            health = wdbx.check_shards_health()
            if not health:
                return await interaction.response.send_message(
                    "No shard configuration available."
                )
            lines = [f"{node}: {'✅' if ok else '❌'}" for node, ok in health.items()]
            await interaction.response.send_message(
                "**Shard Health:**\n" + "\n".join(lines)
            )
        except Exception as e:
            await interaction.response.send_message(f"Shard health error: {e}")

    # Voice channel commands
    @bot.tree.command(name="join", description="Join your voice channel")
    async def slash_join(interaction: discord.Interaction):
        channel = interaction.user.voice.channel if interaction.user.voice else None
        if not channel:
            return await interaction.response.send_message(
                "You are not in a voice channel.", ephemeral=True
            )
        await channel.connect()
        await interaction.response.send_message(f"Joined voice channel {channel.name}")

    @bot.tree.command(name="leave", description="Leave voice channel")
    async def slash_leave(interaction: discord.Interaction):
        voice_client = discord.utils.get(bot.voice_clients, guild=interaction.guild)
        if voice_client and voice_client.is_connected():
            await voice_client.disconnect()
            await interaction.response.send_message("Left voice channel.")
        else:
            await interaction.response.send_message(
                "I am not in a voice channel.", ephemeral=True
            )

    @bot.tree.command(name="say", description="Speak text in voice channel")
    async def slash_say(interaction: discord.Interaction, phrase: str):
        voice_client = discord.utils.get(bot.voice_clients, guild=interaction.guild)
        if not voice_client or not voice_client.is_connected():
            channel = interaction.user.voice.channel if interaction.user.voice else None
            if not channel:
                return await interaction.response.send_message(
                    "You must be in a voice channel.", ephemeral=True
                )
            voice_client = await channel.connect()
        tts = gTTS(text=phrase, lang="en")
        tmpfile = "phrase.mp3"
        tts.save(tmpfile)
        voice_client.play(discord.FFmpegPCMAudio(tmpfile))
        await interaction.response.send_message(f"Speaking: {phrase}", ephemeral=True)

    @bot.tree.command(
        name="screenshot", description="Take a screenshot and send it in chat"
    )
    async def slash_screenshot(interaction: discord.Interaction):
        try:
            screenshot = ImageGrab.grab()
            buf = io.BytesIO()
            screenshot.save(buf, format="PNG")
            buf.seek(0)
            file = discord.File(fp=buf, filename="screenshot.png")
            await interaction.response.send_message("Here's a screenshot:", file=file)
        except Exception as e:
            await interaction.response.send_message(f"Screenshot error: {e}")

    @bot.tree.command(name="app_status", description="Get Flask app status")
    async def slash_app_status(interaction: discord.Interaction):
        try:
            base_url = os.getenv("APP_URL", "http://localhost:8000")
            resp = requests.get(f"{base_url}/health")
            await interaction.response.send_message(
                f"App status: {resp.status_code} - {resp.text}"
            )
        except Exception as e:
            await interaction.response.send_message(f"App status error: {e}")


# Setup slash commands hooks
setup_slash_commands()


def run_discord_bot():
    """
    Start the Discord bot using the token from environment variable.
    """
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        logging.error("Environment variable DISCORD_BOT_TOKEN not set.")
        return
    bot.run(token)
