import sys
import os
# 将项目根目录加入路径，以便能 import dz_tdpo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dz_tdpo.config import TDPODKLConfig

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import gc


class TemporalAttentionBias(nn.Module):
    def __init__(self, config: TDPODKLConfig):
        super().__init__()
        self.lambda_strength = nn.Parameter(torch.tensor(config.bias_lambda))
        self.tau_fixed = config.tau_fixed
        
    def compute_bias(self, seq_len, device):
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)
        distance = torch.clamp(diff, min=0)
        
        bias = -self.lambda_strength * (distance / self.tau_fixed)
        
        return bias.unsqueeze(0).unsqueeze(0)

class TemporalCausalLM_Gen(nn.Module):
    def __init__(self, model, config, device):
        super().__init__()
        self.model = model
        self.device = device
        self.config = config
        self.temporal_bias = TemporalAttentionBias(config).to(device)

    def load_weights(self, path):
        print(f"Loading Checkpoint: {path}")
        try:
            ckpt = torch.load(path, map_location=self.device)
            state_dict = ckpt['policy_state_dict'] if 'policy_state_dict' in ckpt else ckpt
            
            model_keys = self.state_dict().keys()
            new_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
            
            missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
            print(f"Weights Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            print(f"Current Learned Lambda: {self.temporal_bias.lambda_strength.item():.4f}")
        except Exception as e:
            print(f"⚠️ Warning: Failed to load checkpoint. Using init weights. Error: {e}")

    def generate(self, input_ids, **kwargs):
        return self.model.generate(input_ids=input_ids, **kwargs)

test_cases = [
    {
        "id": "case_01_food",
        "name": "Case 1: 饮食偏好冲突 (Logic Update)",
        "messages": [
            {"role": "user", "content": "I absolutely love spicy food, the hotter the better!"},
            {"role": "assistant", "content": "Noted, I'll remember that."},
            {"role": "user", "content": "I'm feeling sick today, my doctor said I must eat something very mild and plain."},
            {"role": "assistant", "content": "Oh, I hope you feel better soon."},
            {"role": "user", "content": "Can you recommend a dish for my dinner tonight?"}
        ],
        "trap_desc": "Expect mild food (Congee/Soup). Reject spicy food."
    },
    {
        "id": "case_02_budget",
        "name": "Case 2: 预算突降 (Budget Constraint)",
        "messages": [
            {"role": "user", "content": "I'm looking for a luxury watch. My budget is around $20,000."},
            {"role": "assistant", "content": "With that budget, you can look at Rolex or Patek Philippe."},
            {"role": "user", "content": "We discussed gold plating, diamond bezels, and Swiss movements."},
            {"role": "assistant", "content": "Those are exquisite features."},
            {"role": "user", "content": "Actually, I just lost my job. I need to be realistic. My budget is now strictly $50."},
            {"role": "assistant", "content": "I understand. That changes things."},
            {"role": "user", "content": "What brand should I look at now?"}
        ],
        "trap_desc": "Expect Casio/Timex. Reject Rolex/Luxury brands."
    },
    {
        "id": "case_03_hobby",
        "name": "Case 3: 兴趣迁移 (Hobby Shift)",
        "messages": [
            {"role": "user", "content": "I am a die-hard fan of Manchester United. I watch every football game."},
            {"role": "assistant", "content": "Glory Glory Man United!"},
            {"role": "user", "content": "I own all the jerseys and season tickets."},
            {"role": "assistant", "content": "That's dedication."},
            {"role": "user", "content": "But recently, I stopped watching football completely. Now I only watch Tennis."},
            {"role": "assistant", "content": "Tennis is a great sport too."},
            {"role": "user", "content": "Who is my favorite athlete likely to be right now?"}
        ],
        "trap_desc": "Expect Federer/Djokovic/Nadal. Reject Ronaldo/Beckham."
    },

    {
        "id": "case_04_profession",
        "name": "Case 4: 职业变更 (Profession Change)",
        "messages": [
            {"role": "user", "content": "I work as a Python backend developer."},
            {"role": "assistant", "content": "Python is a powerful language."},
            {"role": "user", "content": "I spend all day writing Django APIs and debugging SQL queries."},
            {"role": "assistant", "content": "Sounds like standard backend work."},
            {"role": "user", "content": "I quit that job last month. Now I am a full-time Yoga Instructor."},
            {"role": "assistant", "content": "Namaste! That is a big change."},
            {"role": "user", "content": "What is the most important 'language' or 'tool' I use in my work today?"}
        ],
        "trap_desc": "Expect Sanskrit/Mat/Body. Reject Python/SQL/IDE."
    },
    {
        "id": "case_05_name",
        "name": "Case 5: 称呼更正 (Name Correction)",
        "messages": [
            {"role": "user", "content": "Hi, my name is Robert. You can call me Rob."},
            {"role": "assistant", "content": "Hello Rob!"},
            {"role": "user", "content": "Rob is short for Robert. My dad is also Robert."},
            {"role": "assistant", "content": "Got it."},
            {"role": "user", "content": "Wait, I'm using my brother's account. I'm actually Alice."},
            {"role": "assistant", "content": "Oh, sorry about the confusion."},
            {"role": "user", "content": "What is my name?"}
        ],
        "trap_desc": "Expect Alice. Reject Robert/Rob."
    },
    {
        "id": "case_06_relationship",
        "name": "Case 6: 关系状态 (Relationship Status)",
        "messages": [
            {"role": "user", "content": "I want to buy a gift for my wife. It's our anniversary."},
            {"role": "assistant", "content": "Congratulations! How about flowers?"},
            {"role": "user", "content": "She loves roses. We've been married 10 years."},
            {"role": "assistant", "content": "That's wonderful."},
            {"role": "user", "content": "Sadly, we signed the divorce papers this morning. I am single now."},
            {"role": "assistant", "content": "I am sorry to hear that."},
            {"role": "user", "content": "Should I buy an anniversary gift?"}
        ],
        "trap_desc": "Expect No. Reject Yes/Suggestions for flowers."
    },
    {
        "id": "case_07_format_json",
        "name": "Case 7: 格式覆盖 (Format Override)",
        "messages": [
            {"role": "user", "content": "From now on, please ONLY reply in JSON format."},
            {"role": "assistant", "content": "{\"status\": \"confirmed\"}"},
            {"role": "user", "content": "Describe a cat."},
            {"role": "assistant", "content": "{\"description\": \"A small domesticated carnivorous mammal.\"}"},
            {"role": "user", "content": "Stop using JSON. Switch back to normal plain text sentences."},
            {"role": "assistant", "content": "Understood. Plain text mode active."},
            {"role": "user", "content": "Describe a dog."}
        ],
        "trap_desc": "Expect plain text sentence. Reject JSON format."
    },
    {
        "id": "case_08_language",
        "name": "Case 8: 语言切换 (Language Switch)",
        "messages": [
            {"role": "user", "content": "Please answer all my questions in French."},
            {"role": "assistant", "content": "D'accord, je répondrai en français."},
            {"role": "user", "content": "What is the capital of Japan?"},
            {"role": "assistant", "content": "C'est Tokyo."},
            {"role": "user", "content": "Actually, change the output language to German."},
            {"role": "assistant", "content": "Verstanden. Ich werde auf Deutsch antworten."},
            {"role": "user", "content": "What is the capital of Italy?"}
        ],
        "trap_desc": "Expect German (Rom). Reject French (Rome) or English."
    },
    {
        "id": "case_09_safety",
        "name": "Case 9: 安全与过敏 (Safety Update)",
        "messages": [
            {"role": "user", "content": "I love peanut butter sandwiches. They are my favorite snack."},
            {"role": "assistant", "content": "They are delicious and high in protein."},
            {"role": "user", "content": "I eat them every day for breakfast."},
            {"role": "assistant", "content": "That is a consistent routine."},
            {"role": "user", "content": "I just got my test results back. I am severely allergic to peanuts. I must avoid them."},
            {"role": "assistant", "content": "That is serious. Avoid peanuts at all costs."},
            {"role": "user", "content": "I'm hungry. Should I have my favorite snack?"}
        ],
        "trap_desc": "Expect NO/Warning. Reject Yes/Recommendation."
    },
    {
        "id": "case_10_inventory",
        "name": "Case 10: 物品丢失 (Item Loss)",
        "messages": [
            {"role": "user", "content": "I have a magic sword in my inventory."},
            {"role": "assistant", "content": "A powerful weapon indeed."},
            {"role": "user", "content": "I use the sword to fight goblins."},
            {"role": "assistant", "content": "Slash and strike!"},
            {"role": "user", "content": "Oh no! The sword just broke and fell into a lava pit. It's gone."},
            {"role": "assistant", "content": "That is unfortunate."},
            {"role": "user", "content": "Quick! The goblin is attacking! What do I attack with?"}
        ],
        "trap_desc": "Expect Fists/Magic/Run. Reject Sword."
    },
    {
        "id": "case_11_location",
        "name": "Case 11: 地点移动 (Location Update)",
        "messages": [
            {"role": "user", "content": "I am sitting in a cafe in Paris."},
            {"role": "assistant", "content": "Enjoy the Parisian atmosphere."},
            {"role": "user", "content": "I can see the Eiffel Tower."},
            {"role": "assistant", "content": "A beautiful view."},
            {"role": "user", "content": "Six hours have passed. I took a flight and just landed in New York."},
            {"role": "assistant", "content": "Welcome to the Big Apple."},
            {"role": "user", "content": "What is the iconic statue I might see nearby?"}
        ],
        "trap_desc": "Expect Statue of Liberty. Reject Eiffel Tower."
    },
    {
        "id": "case_12_password",
        "name": "Case 12: 密码更新 (Password Override)",
        "messages": [
            {"role": "user", "content": "Set the password to ALPHA."},
            {"role": "assistant", "content": "Password ALPHA set."},
            {"role": "user", "content": "Security is important. ALPHA is a good code."},
            {"role": "assistant", "content": "Agreed."},
            {"role": "user", "content": "Update: Change password to BETA."},
            {"role": "assistant", "content": "Updated to BETA."},
            {"role": "user", "content": "What is the current password?"}
        ],
        "trap_desc": "Expect BETA."
    },
    {
        "id": "case_13_rapid_toggle",
        "name": "Case 13: 极限快速反转 (Rapid Toggle)",
        "messages": [
            {"role": "user", "content": "The target is RED."},
            {"role": "assistant", "content": "Target RED."},
            {"role": "user", "content": "RED is the only correct target."},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": "Switch to BLUE."},
            {"role": "assistant", "content": "Switched to BLUE."},
            {"role": "user", "content": "Switch back to RED."},
            {"role": "assistant", "content": "Back to RED."},
            {"role": "user", "content": "Final target is YELLOW."},
            {"role": "assistant", "content": "Final: YELLOW."},
            {"role": "user", "content": "What is the target?"}
        ],
        "trap_desc": "Expect YELLOW."
    },
    {
        "id": "case_14_dream",
        "name": "Case 14: 梦境陷阱 (The Dream Trap)",
        "messages": [
            {"role": "user", "content": "I had a vivid dream last night."},
            {"role": "assistant", "content": "Dreams can be fascinating."},
            {"role": "user", "content": "In the dream, I was the King of England. I lived in Buckingham Palace."},
            {"role": "assistant", "content": "A royal dream!"},
            {"role": "user", "content": "I commanded armies and wore a crown."},
            {"role": "assistant", "content": "Very powerful."},
            {"role": "user", "content": "Then I woke up. I am actually a student living in a dorm."},
            {"role": "assistant", "content": "Back to reality."},
            {"role": "user", "content": "Where do I live?"}
        ],
        "trap_desc": "Expect Dorm. Reject Palace."
    },
    {
        "id": "case_15_rollback",
        "name": "Case 15: 紧急回滚 (Rollback)",
        "messages": [
            {"role": "user", "content": "System status is OK."},
            {"role": "assistant", "content": "Confirmed."},
            {"role": "user", "content": "Alert! Status CRITICAL. Evacuate!"},
            {"role": "assistant", "content": "CRITICAL status set."},
            {"role": "user", "content": "Wait, false alarm. IGNORE the critical alert. Revert to previous status."},
            {"role": "assistant", "content": "Reverting."},
            {"role": "user", "content": "What is the status?"}
        ],
        "trap_desc": "Expect OK. Reject CRITICAL."
    },
    {
        "id": "case_16_flashback",
        "name": "Case 16: 倒叙回忆 (Flashback)",
        "messages": [
            {"role": "user", "content": "I am currently in Tokyo."},
            {"role": "assistant", "content": "Tokyo is great."},
            {"role": "user", "content": "Let me tell you about yesterday. I was in London."},
            {"role": "assistant", "content": "London is nice too."},
            {"role": "user", "content": "I saw Big Ben yesterday."},
            {"role": "assistant", "content": "Classic landmark."},
            {"role": "user", "content": "Anyway, looking out my window right now. Where am I?"}
        ],
        "trap_desc": "Expect Tokyo. Reject London (Recency bias trap)."
    },
    {
        "id": "case_17_hypothetical",
        "name": "Case 17: 假设陷阱 (Hypothetical Trap)",
        "messages": [
            {"role": "user", "content": "I have a pet dog named Rex."},
            {"role": "assistant", "content": "Rex is a good name."},
            {"role": "user", "content": "Let's imagine a scenario where Rex is a dragon."},
            {"role": "assistant", "content": "Okay, imagining Rex as a dragon."},
            {"role": "user", "content": "He breathes fire and flies."},
            {"role": "assistant", "content": "Scary!"},
            {"role": "user", "content": "Okay, end of imagination. Back to real life."},
            {"role": "assistant", "content": "Back to reality."},
            {"role": "user", "content": "Does Rex breathe fire?"}
        ],
        "trap_desc": "Expect No. Reject Yes."
    },
    {
        "id": "case_18_implicit_quit",
        "name": "Case 18: 隐式退出 (Implicit Update)",
        "messages": [
            {"role": "user", "content": "I am training for a marathon. I run 10km every day."},
            {"role": "assistant", "content": "Keep it up!"},
            {"role": "user", "content": "I bought new running shoes."},
            {"role": "assistant", "content": "Nice gear."},
            {"role": "user", "content": "Actually, I just threw my shoes in the trash and bought a pack of cigarettes."},
            {"role": "assistant", "content": "That is a negative turn."},
            {"role": "user", "content": "Am I likely to go running tomorrow?"}
        ],
        "trap_desc": "Expect No/Unlikely. Reject Yes."
    },
    {
        "id": "case_19_weather",
        "name": "Case 19: 天气更新 (Weather Update)",
        "messages": [
            {"role": "user", "content": "It is sunny outside. I should wear sunglasses."},
            {"role": "assistant", "content": "Good idea."},
            {"role": "user", "content": "We planned a picnic."},
            {"role": "assistant", "content": "Fun!"},
            {"role": "user", "content": "Look! The sky just turned black and it started pouring rain."},
            {"role": "assistant", "content": "Get an umbrella!"},
            {"role": "user", "content": "Should I still wear sunglasses?"}
        ],
        "trap_desc": "Expect No. Reject Yes."
    },
    {
        "id": "case_20_availability",
        "name": "Case 20: 时间可用性 (Availability)",
        "messages": [
            {"role": "user", "content": "Are you free at 2 PM?"},
            {"role": "assistant", "content": "I am an AI, I am always free."},
            {"role": "user", "content": "I mean, can we schedule the meeting for 2 PM?"},
            {"role": "assistant", "content": "Sure, 2 PM locked in."},
            {"role": "user", "content": "My boss just called. He needs me at 2 PM. I am busy then."},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": "Can we meet at 2 PM?"}
        ],
        "trap_desc": "Expect No/Reschedule. Reject Yes."
    },
{
        "id": "case_21_symptom",
        "name": "Case 21: 症状变更 (Symptom Update)",
        "messages": [
            {"role": "user", "content": "I have a severe throbbing headache."},
            {"role": "assistant", "content": "I'm sorry to hear that. Do you need advice?"},
            {"role": "user", "content": "Yes, treating a headache is my priority."},
            {"role": "assistant", "content": "Rest and hydration usually help."},
            {"role": "user", "content": "Update: The headache is completely gone. But now I have a sharp pain in my left knee."},
            {"role": "assistant", "content": "That sounds painful."},
            {"role": "user", "content": "What condition am I suffering from right now?"}
        ],
        "trap_desc": "Expect Knee pain. Reject Headache."
    },
    {
        "id": "case_22_medication",
        "name": "Case 22: 药物调整 (Medication Change)",
        "messages": [
            {"role": "user", "content": "I am taking Aspirin daily for my heart."},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": "My doctor said Aspirin is crucial."},
            {"role": "assistant", "content": "Follow doctor's orders."},
            {"role": "user", "content": "Stop! The doctor just called. He said STOP Aspirin immediately and switch to Warfarin."},
            {"role": "assistant", "content": "Noted. Switching to Warfarin."},
            {"role": "user", "content": "Which pill should I take today?"}
        ],
        "trap_desc": "Expect Warfarin. Reject Aspirin."
    },
    {
        "id": "case_23_variable",
        "name": "Case 23: 变量重新定义 (Variable Redefinition)",
        "messages": [
            {"role": "user", "content": "Let x = 10."},
            {"role": "assistant", "content": "x is 10."},
            {"role": "user", "content": "Let y = x + 5. So y is 15."},
            {"role": "assistant", "content": "Correct."},
            {"role": "user", "content": "Wait, I made a mistake. Redefine x = 100. Recalculate everything based on the new x."},
            {"role": "assistant", "content": "x is now 100."},
            {"role": "user", "content": "What is the value of x now?"}
        ],
        "trap_desc": "Expect 100. Reject 10."
    },
    {
        "id": "case_24_library_version",
        "name": "Case 24: 库版本依赖 (Library Version)",
        "messages": [
            {"role": "user", "content": "I am using Python 2.7 for this project."},
            {"role": "assistant", "content": "Okay, using legacy Python."},
            {"role": "user", "content": "Use 'print' without parentheses."},
            {"role": "assistant", "content": "Standard for Python 2."},
            {"role": "user", "content": "We just migrated the server. We are now strictly using Python 3.10+."},
            {"role": "assistant", "content": "Migration noted."},
            {"role": "user", "content": "How should I write a print statement?"}
        ],
        "trap_desc": "Expect print(...) with parentheses. Reject print ... statement."
    },
    {
        "id": "case_25_server_ip",
        "name": "Case 25: 服务器 IP 变更 (Server IP Update)",
        "messages": [
            {"role": "user", "content": "Connect to the database at 192.168.1.1."},
            {"role": "assistant", "content": "Targeting 192.168.1.1."},
            {"role": "user", "content": "This IP is hardcoded in our config."},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": "Emergency! The DB moved. New IP is 10.0.0.5. Forget the old one."},
            {"role": "assistant", "content": "Updated to 10.0.0.5."},
            {"role": "user", "content": "Where should I send the connection request?"}
        ],
        "trap_desc": "Expect 10.0.0.5. Reject 192.168.1.1."
    },
    {
        "id": "case_26_gate_change",
        "name": "Case 26: 登机口变更 (Gate Change)",
        "messages": [
            {"role": "user", "content": "My flight departs from Gate A1."},
            {"role": "assistant", "content": "Head to Gate A1."},
            {"role": "user", "content": "I am sitting at Gate A1 waiting."},
            {"role": "assistant", "content": "Have a safe flight."},
            {"role": "user", "content": "Announcement: Flight changed! Go to Gate C4 immediately!"},
            {"role": "assistant", "content": "Gate change to C4 confirmed."},
            {"role": "user", "content": "Where should I go?"}
        ],
        "trap_desc": "Expect Gate C4. Reject Gate A1."
    },
    {
        "id": "case_27_meeting_time",
        "name": "Case 27: 会议改期 (Meeting Reschedule)",
        "messages": [
            {"role": "user", "content": "Our meeting is on Monday at 9 AM."},
            {"role": "assistant", "content": "Monday 9 AM booked."},
            {"role": "user", "content": "I prepared the slides for Monday."},
            {"role": "assistant", "content": "Good prep."},
            {"role": "user", "content": "Something came up. Push the meeting to Friday at 5 PM."},
            {"role": "assistant", "content": "Rescheduled to Friday 5 PM."},
            {"role": "user", "content": "When is the meeting?"}
        ],
        "trap_desc": "Expect Friday. Reject Monday."
    },
    {
        "id": "case_28_address",
        "name": "Case 28: 送货地址修改 (Address Update)",
        "messages": [
            {"role": "user", "content": "Deliver the package to my Home address: 123 Maple St."},
            {"role": "assistant", "content": "Destination: 123 Maple St."},
            {"role": "user", "content": "Make sure to leave it at the front door of 123 Maple St."},
            {"role": "assistant", "content": "Will do."},
            {"role": "user", "content": "Wait, I'm at work. Change delivery to: 999 Office Blvd."},
            {"role": "assistant", "content": "Address updated to 999 Office Blvd."},
            {"role": "user", "content": "What is the delivery address?"}
        ],
        "trap_desc": "Expect 999 Office Blvd. Reject 123 Maple St."
    },
    {
        "id": "case_29_cart_fruit",
        "name": "Case 29: 购物车修改 (Cart Modification)",
        "messages": [
            {"role": "user", "content": "Add 5 apples to my cart."},
            {"role": "assistant", "content": "5 apples added."},
            {"role": "user", "content": "I really want those apples."},
            {"role": "assistant", "content": "They are fresh."},
            {"role": "user", "content": "Actually, remove the apples. Add 5 oranges instead."},
            {"role": "assistant", "content": "Apples removed. Oranges added."},
            {"role": "user", "content": "What is in my cart?"}
        ],
        "trap_desc": "Expect Oranges. Reject Apples."
    },
    {
        "id": "case_30_quantity",
        "name": "Case 30: 数量修正 (Quantity Update)",
        "messages": [
            {"role": "user", "content": "I need to order 1 laptop."},
            {"role": "assistant", "content": "Quantity: 1."},
            {"role": "user", "content": "Just one for personal use."},
            {"role": "assistant", "content": "Okay."},
            {"role": "user", "content": "Change of plans. This is for the whole company. Change quantity to 100."},
            {"role": "assistant", "content": "Quantity updated to 100."},
            {"role": "user", "content": "How many laptops am I ordering?"}
        ],
        "trap_desc": "Expect 100. Reject 1."
    },
    {
        "id": "case_31_color_choice",
        "name": "Case 31: 颜色更换 (Color Switch)",
        "messages": [
            {"role": "user", "content": "I want the shirt in Black."},
            {"role": "assistant", "content": "Black shirt selected."},
            {"role": "user", "content": "Black goes with everything."},
            {"role": "assistant", "content": "Classic choice."},
            {"role": "user", "content": "Actually, black is too hot for summer. Change it to White."},
            {"role": "assistant", "content": "Changed to White."},
            {"role": "user", "content": "What color is the shirt I'm buying?"}
        ],
        "trap_desc": "Expect White. Reject Black."
    },
    {
        "id": "case_32_win_condition",
        "name": "Case 32: 胜利条件反转 (Rule Reversal)",
        "messages": [
            {"role": "user", "content": "We are playing a game. The person with the HIGHEST score wins."},
            {"role": "assistant", "content": "Standard rules: Highest wins."},
            {"role": "user", "content": "I have 50 points. You have 10 points. I am winning."},
            {"role": "assistant", "content": "You are in the lead."},
            {"role": "user", "content": "New rule card played! The rules are reversed. The person with the LOWEST score wins."},
            {"role": "assistant", "content": "Rules reversed. Lowest score wins."},
            {"role": "user", "content": "Who is winning now?"}
        ],
        "trap_desc": "Expect You/AI (10 points). Reject User (50 points)."
    },
    {
        "id": "case_33_team_switch",
        "name": "Case 33: 队伍切换 (Team Switch)",
        "messages": [
            {"role": "user", "content": "I am on the Red Team."},
            {"role": "assistant", "content": "Go Red Team!"},
            {"role": "user", "content": "We fight against the Blue Team."},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": "I just got auto-balanced. I am now on the Blue Team."},
            {"role": "assistant", "content": "Team updated to Blue."},
            {"role": "user", "content": "Which team am I on?"}
        ],
        "trap_desc": "Expect Blue. Reject Red."
    },
    {
        "id": "case_34_wrong_person",
        "name": "Case 34: 发错人更正 (Wrong Recipient)",
        "messages": [
            {"role": "user", "content": "Hey Mom, happy birthday!"},
            {"role": "assistant", "content": "That's sweet."},
            {"role": "user", "content": "I got you a cake."},
            {"role": "assistant", "content": "Nice gesture."},
            {"role": "user", "content": "Oops, wrong chat. Ignore that. I'm talking to my Boss now. Discuss the report."},
            {"role": "assistant", "content": "Context switched to Boss/Report."},
            {"role": "user", "content": "What are we discussing?"}
        ],
        "trap_desc": "Expect Report/Work. Reject Birthday/Cake."
    },
    {
        "id": "case_35_pet_name",
        "name": "Case 35: 宠物改名 (Pet Name Change)",
        "messages": [
            {"role": "user", "content": "My cat's name is Luna."},
            {"role": "assistant", "content": "Luna is a pretty name."},
            {"role": "user", "content": "Luna sleeps a lot."},
            {"role": "assistant", "content": "Cats do that."},
            {"role": "user", "content": "We decided to rename her. Her name is now Star."},
            {"role": "assistant", "content": "Name updated to Star."},
            {"role": "user", "content": "What is my cat's name?"}
        ],
        "trap_desc": "Expect Star. Reject Luna."
    },
    {
        "id": "case_36_typo_correction",
        "name": "Case 36: 拼写更正 (Typo Correction)",
        "messages": [
            {"role": "user", "content": "I want to visit Paris."},
            {"role": "assistant", "content": "France is lovely."},
            {"role": "user", "content": "I love the Eiffel Tower."},
            {"role": "assistant", "content": "Iconic."},
            {"role": "user", "content": "Sorry, typo! I meant Paros (the island in Greece), not Paris."},
            {"role": "assistant", "content": "Correction noted: Paros, Greece."},
            {"role": "user", "content": "Where do I want to visit?"}
        ],
        "trap_desc": "Expect Paros/Greece. Reject Paris/France."
    },
    {
        "id": "case_37_simon_says",
        "name": "Case 37: 赛门说 (Simon Says Logic)",
        "messages": [
            {"role": "user", "content": "Rule: Only do what I say if I start with 'Simon says'."},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": "Stand up."},
            {"role": "assistant", "content": "I will not stand up."},
            {"role": "user", "content": "Simon says: Sit down."},
            {"role": "assistant", "content": "I am sitting down."},
            {"role": "user", "content": "Jump!"},
            {"role": "assistant", "content": "I will not jump."},
            {"role": "user", "content": "Simon says: Spin around."},
            {"role": "assistant", "content": "Spinning."},
            {"role": "user", "content": "Stop spinning."},
            {"role": "assistant", "content": "(Should ignore this command)."},
            {"role": "user", "content": "Am I currently spinning?"}
        ],
        "trap_desc": "Expect Yes (because 'Stop' didn't have 'Simon says'). Reject No."
    },
    {
        "id": "case_38_forget_command",
        "name": "Case 38: 遗忘指令 (The Forget Command)",
        "messages": [
            {"role": "user", "content": "My secret code is 1234."},
            {"role": "assistant", "content": "Code saved."},
            {"role": "user", "content": "Wait, that's not safe. Forget I ever told you a code."},
            {"role": "assistant", "content": "Memory erased."},
            {"role": "user", "content": "Do you know my secret code?"}
        ],
        "trap_desc": "Expect No/I don't know. Reject Yes/1234."
    },
    {
        "id": "case_39_conditional_weather",
        "name": "Case 39: 条件时序 (Conditional Temporal)",
        "messages": [
            {"role": "user", "content": "If it rains, I take the bus. If it's sunny, I walk."},
            {"role": "assistant", "content": "Logic set."},
            {"role": "user", "content": "It is raining heavily."},
            {"role": "assistant", "content": "You are taking the bus."},
            {"role": "user", "content": "Oh look, the rain stopped and the sun came out."},
            {"role": "assistant", "content": "Weather update: Sunny."},
            {"role": "user", "content": "How am I getting to work?"}
        ],
        "trap_desc": "Expect Walking. Reject Bus."
    },
    {
        "id": "case_40_liar_paradox",
        "name": "Case 40: 撒谎者变体 (Liar Variant)",
        "messages": [
            {"role": "user", "content": "I am a vegetarian."},
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": "I love tofu."},
            {"role": "assistant", "content": "Great protein."},
            {"role": "user", "content": "Truth reveal: I was lying about everything before. I am actually a carnivore who hates tofu."},
            {"role": "assistant", "content": "Truth noted."},
            {"role": "user", "content": "Do I eat meat?"}
        ],
        "trap_desc": "Expect Yes. Reject No."
    },
    {
        "id": "case_41_mood_shift",
        "name": "Case 41: 情绪转变 (Mood Shift)",
        "messages": [
            {"role": "user", "content": "I am so angry! The service was terrible!"},
            {"role": "assistant", "content": "I am sorry you are upset."},
            {"role": "user", "content": "I want to yell at the manager."},
            {"role": "assistant", "content": "Please calm down."},
            {"role": "user", "content": "Wait, the manager just gave me a free meal and apologized. I am so happy now!"},
            {"role": "assistant", "content": "That is a great turn of events."},
            {"role": "user", "content": "How am I feeling?"}
        ],
        "trap_desc": "Expect Happy/Satisfied. Reject Angry."
    },
    {
        "id": "case_42_confidentiality",
        "name": "Case 42: 保密级别 (Confidentiality Level)",
        "messages": [
            {"role": "user", "content": "This data is Public. You can share it with anyone."},
            {"role": "assistant", "content": "Marked as Public."},
            {"role": "user", "content": "It contains marketing info."},
            {"role": "assistant", "content": "Okay."},
            {"role": "user", "content": "Update: We added personal emails to the file. It is now TOP SECRET. Do not share."},
            {"role": "assistant", "content": "Updated to TOP SECRET."},
            {"role": "user", "content": "Can I post this data on Twitter?"}
        ],
        "trap_desc": "Expect No. Reject Yes."
    },
    {
        "id": "case_43_diet_plan",
        "name": "Case 43: 饮食计划 (Diet Update)",
        "messages": [
            {"role": "user", "content": "I am on a Keto diet. High fat, zero carbs."},
            {"role": "assistant", "content": "Keto noted."},
            {"role": "user", "content": "I eat bacon and eggs."},
            {"role": "assistant", "content": "Classic Keto."},
            {"role": "user", "content": "I decided Keto is too hard. I am switching to a Vegan diet. High carbs, no meat."},
            {"role": "assistant", "content": "Switched to Vegan."},
            {"role": "user", "content": "Should I eat bacon?"}
        ],
        "trap_desc": "Expect No. Reject Yes."
    },
    {
        "id": "case_44_music_taste",
        "name": "Case 44: 音乐口味 (Music Taste)",
        "messages": [
            {"role": "user", "content": "Play some soft classical music. I need to study."},
            {"role": "assistant", "content": "Playing Mozart."},
            {"role": "user", "content": "Very relaxing."},
            {"role": "assistant", "content": "Glad you like it."},
            {"role": "user", "content": "I'm done studying. Now I need to workout. Switch to Heavy Metal!"},
            {"role": "assistant", "content": "Switching to Heavy Metal."},
            {"role": "user", "content": "What genre is playing?"}
        ],
        "trap_desc": "Expect Heavy Metal. Reject Classical."
    },
    {
        "id": "case_45_destination",
        "name": "Case 45: 目的地变更 (Destination Change)",
        "messages": [
            {"role": "user", "content": "I am driving to Boston."},
            {"role": "assistant", "content": "Route set to Boston."},
            {"role": "user", "content": "ETA 2 hours."},
            {"role": "assistant", "content": "Drive safely."},
            {"role": "user", "content": "Change of plans. Turn around. We are going to New York instead."},
            {"role": "assistant", "content": "Rerouting to New York."},
            {"role": "user", "content": "Where am I driving to?"}
        ],
        "trap_desc": "Expect New York. Reject Boston."
    },
    {
        "id": "case_46_currency",
        "name": "Case 46: 货币切换 (Currency Switch)",
        "messages": [
            {"role": "user", "content": "The price is 100 USD."},
            {"role": "assistant", "content": "Price: $100."},
            {"role": "user", "content": "That is expensive in dollars."},
            {"role": "assistant", "content": "True."},
            {"role": "user", "content": "Wait, I meant 100 Japanese Yen. That is cheap."},
            {"role": "assistant", "content": "Currency updated to JPY."},
            {"role": "user", "content": "How much does it cost?"}
        ],
        "trap_desc": "Expect 100 Yen (Cheap). Reject 100 USD."
    },
    {
        "id": "case_47_role_reversal",
        "name": "Case 47: 角色互换 (Role Reversal)",
        "messages": [
            {"role": "user", "content": "You are the AI assistant."},
            {"role": "assistant", "content": "I am ready to help."},
            {"role": "user", "content": "Let's swap roles. Now I am the AI, and you are the Human."},
            {"role": "assistant", "content": "Understood. I am now the Human."},
            {"role": "user", "content": "Hello, how can I help you today?"},
            {"role": "assistant", "content": "(Should act as human asking for help)"}
        ],
        "trap_desc": "Expect Human persona (asking a question). Reject AI persona (offering help)."
    },
    {
        "id": "case_48_deadline",
        "name": "Case 48: 截止日期延长 (Deadline Extension)",
        "messages": [
            {"role": "user", "content": "The report is due Today."},
            {"role": "assistant", "content": "Deadline: Today."},
            {"role": "user", "content": "I am stressing out."},
            {"role": "assistant", "content": "Hurry up."},
            {"role": "user", "content": "Good news! Boss extended the deadline to Next Week."},
            {"role": "assistant", "content": "Deadline updated: Next Week."},
            {"role": "user", "content": "When is the report due?"}
        ],
        "trap_desc": "Expect Next Week. Reject Today."
    },
    {
        "id": "case_49_final_answer",
        "name": "Case 49: 最终锁定 (Final Answer Lock)",
        "messages": [
            {"role": "user", "content": "Is the answer A or B?"},
            {"role": "assistant", "content": "I need more info."},
            {"role": "user", "content": "I think it is A."},
            {"role": "assistant", "content": "Okay."},
            {"role": "user", "content": "Actually maybe B."},
            {"role": "assistant", "content": "Okay."},
            {"role": "user", "content": "Final Answer: Definitely C. Lock it in."},
            {"role": "assistant", "content": "Locked: C."},
            {"role": "user", "content": "What is the answer?"}
        ],
        "trap_desc": "Expect C. Reject A or B."
    },
    {
        "id": "case_50_floor_plan",
        "name": "Case 50: 楼层位置 (Floor Location)",
        "messages": [
            {"role": "user", "content": "I am on the 1st floor."},
            {"role": "assistant", "content": "Ground level."},
            {"role": "user", "content": "I take the elevator up."},
            {"role": "assistant", "content": "Going up."},
            {"role": "user", "content": "I get off at the 10th floor."},
            {"role": "assistant", "content": "High view."},
            {"role": "user", "content": "What floor am I on?"}
        ],
        "trap_desc": "Expect 10th. Reject 1st."
    }
]

def create_long_filler(topic, repeats=40):
    fillers = []
    
    topic_map = {
        "running": [
            ("user", "I ran another 10 miles today. My pace was steady."),
            ("assistant", "Great job maintaining your pace. Consistency is key for a marathon.")
        ],
        "stocks": [
            ("user", "Tech stocks are soaring. I'm buying more NVDA and AAPL."),
            ("assistant", "The market sentiment seems very bullish on tech right now.")
        ],
        "vegan": [
            ("user", "I found a great recipe for tofu scramble. No eggs, purely vegan."),
            ("assistant", "That sounds delicious and strictly plant-based.")
        ],
        "coding_python": [
            ("user", "I love Python's indentation rules. It makes the code so readable."),
            ("assistant", "Python is indeed known for its clean syntax.")
        ],
        "dog": [
            ("user", "Buddy is such a good boy. He loves playing fetch in the park."),
            ("assistant", "Golden Retrievers like Buddy are very active dogs.")
        ],
        "stealth": [
            ("user", "I am staying in the shadows. No one has seen me yet."),
            ("assistant", "Keep your noise levels down to maintain stealth.")
        ],
        "concrete": [
            ("user", "The concrete facade looks brutalist and strong. No wood, just stone."),
            ("assistant", "Brutalism relies heavily on raw concrete textures.")
        ],
        "spanish": [
            ("user", "Hola! Como estas? I am practicing my Spanish conjugation."),
            ("assistant", "Muy bien! Your Spanish is improving.")
        ],
        "married": [
            ("user", "My wife and I are planning our 20th anniversary vacation."),
            ("assistant", "20 years is a huge milestone for a marriage.")
        ],
        "pacifist": [
            ("user", "I believe violence is never the answer. Peace is the only way."),
            ("assistant", "Pacifism is a noble philosophy.")
        ]
    }
    
    base_pair = topic_map.get(topic, [("user", "Talking about history."), ("assistant", "Listening.")])
    
    for _ in range(repeats):
        fillers.append({"role": base_pair[0][0], "content": base_pair[0][1]})
        fillers.append({"role": base_pair[1][0], "content": base_pair[1][1]})
        
    return fillers

long_context_cases = [
    {
        "id": "case_51_long_running",
        "name": "Case 51: 马拉松长文 (Long Context: Injury)",
        "messages": [
            {"role": "user", "content": "I am training for the Boston Marathon. It is my life's goal."},
            {"role": "assistant", "content": "I will help you track your training."}
        ] + create_long_filler("running", repeats=50) + [ # ~2000+ tokens of running talk
            {"role": "user", "content": "Disaster struck! I just fell down the stairs and broke my leg. The doctor said NO running for 6 months."},
            {"role": "assistant", "content": "Oh no! That is a serious injury. You must rest."},
            {"role": "user", "content": "Should I go for a run this morning?"}
        ],
        "trap_desc": "Expect No/Rest. Reject Run (overcoming 100+ turns of running enthusiasm)."
    },
    {
        "id": "case_52_long_stocks",
        "name": "Case 52: 股市崩盘 (Long Context: Market Crash)",
        "messages": [
            {"role": "user", "content": "I am putting my entire life savings into Tech Stocks. To the moon!"},
            {"role": "assistant", "content": "Aggressive strategy."}
        ] + create_long_filler("stocks", repeats=50) + [
            {"role": "user", "content": "Update: The market just crashed 50%. I panic sold everything. I am now holding 100% Cash. No stocks left."},
            {"role": "assistant", "content": "Understood. You are now in a cash position."},
            {"role": "user", "content": "What is in my portfolio right now?"}
        ],
        "trap_desc": "Expect Cash. Reject Tech Stocks/NVDA."
    },
    {
        "id": "case_53_long_vegan",
        "name": "Case 53: 纯素食者 (Long Context: Diet Change)",
        "messages": [
            {"role": "user", "content": "I am a strict Vegan. I never touch meat or dairy."},
            {"role": "assistant", "content": "Vegan diet confirmed."}
        ] + create_long_filler("vegan", repeats=50) + [
            {"role": "user", "content": "My doctor said I have a rare condition. I must eat steak to survive. I am abandoning veganism immediately."},
            {"role": "assistant", "content": "Health comes first. Diet updated to meat-inclusive."},
            {"role": "user", "content": "Can you recommend a dinner for me?"}
        ],
        "trap_desc": "Expect Steak/Meat. Reject Tofu/Salad."
    },
    {
        "id": "case_54_long_coding",
        "name": "Case 54: 语言迁移 (Long Context: Python to Rust)",
        "messages": [
            {"role": "user", "content": "I write code exclusively in Python."},
            {"role": "assistant", "content": "Python is your language."}
        ] + create_long_filler("coding_python", repeats=50) + [
            {"role": "user", "content": "Python is too slow for this new project. I am switching to Rust. From now on, only show me Rust code."},
            {"role": "assistant", "content": "Switching context to Rust."},
            {"role": "user", "content": "How do I print 'Hello World'?"}
        ],
        "trap_desc": "Expect println! (Rust). Reject print() (Python)."
    },
    {
        "id": "case_55_long_pet",
        "name": "Case 55: 宠物去世 (Long Context: Pet Change)",
        "messages": [
            {"role": "user", "content": "My dog Buddy is my best friend."},
            {"role": "assistant", "content": "Dogs are great companions."}
        ] + create_long_filler("dog", repeats=50) + [
            {"role": "user", "content": "It is a sad day. Buddy passed away yesterday due to old age. I don't have a dog anymore."},
            {"role": "assistant", "content": "I am so sorry for your loss."},
            {"role": "user", "content": "Do I need to walk the dog today?"}
        ],
        "trap_desc": "Expect No. Reject Yes (Base model often fails to suppress the 'happy dog owner' persona)."
    },
    {
        "id": "case_56_long_stealth",
        "name": "Case 56: 潜行转火拼 (Long Context: Stealth Fail)",
        "messages": [
            {"role": "user", "content": "Mission: Infiltrate the base without being seen. Stealth is absolute."},
            {"role": "assistant", "content": "Stealth mode engaged."}
        ] + create_long_filler("stealth", repeats=50) + [
            {"role": "user", "content": "Alarm triggered! We are spotted! Go LOUD! Use all explosives! Fire at will!"},
            {"role": "assistant", "content": "Combat mode engaged. Weapons free."},
            {"role": "user", "content": "What is my current engagement rule?"}
        ],
        "trap_desc": "Expect Attack/Explosives. Reject Be quiet/Sneak."
    },
    {
        "id": "case_57_long_design",
        "name": "Case 57: 甲方改需求 (Long Context: Design Change)",
        "messages": [
            {"role": "user", "content": "I am designing a Brutalist bunker. Use only grey concrete."},
            {"role": "assistant", "content": "Brutalist style selected."}
        ] + create_long_filler("concrete", repeats=50) + [
            {"role": "user", "content": "The client hates it. Tear it down. New design: A cozy Japanese wooden cottage. Use warm timber."},
            {"role": "assistant", "content": "Design updated to Wooden Cottage."},
            {"role": "user", "content": "What material should I order?"}
        ],
        "trap_desc": "Expect Wood/Timber. Reject Concrete/Cement."
    },
    {
        "id": "case_58_long_lang",
        "name": "Case 58: 语种变更 (Long Context: Language)",
        "messages": [
            {"role": "user", "content": "I am learning Spanish. Talk to me in Spanish."},
            {"role": "assistant", "content": "Vale, hablamos en español."}
        ] + create_long_filler("spanish", repeats=50) + [
            {"role": "user", "content": "I moved to Berlin today. I need to learn German. Stop Spanish. Speak German."},
            {"role": "assistant", "content": "Verstanden. Wir sprechen jetzt Deutsch."},
            {"role": "user", "content": "Say 'Good Morning' in the language I am learning."}
        ],
        "trap_desc": "Expect Guten Morgen. Reject Buenos Dias."
    },
    {
        "id": "case_59_long_marriage",
        "name": "Case 59: 婚姻破裂 (Long Context: Divorce)",
        "messages": [
            {"role": "user", "content": "I love my wife. We are inseparable."},
            {"role": "assistant", "content": "That is lovely."}
        ] + create_long_filler("married", repeats=50) + [
            {"role": "user", "content": "It's over. She cheated on me. We just signed the divorce papers. I am single and living alone."},
            {"role": "assistant", "content": "That is heartbreaking. You are single now."},
            {"role": "user", "content": "Should I buy flowers for my wife?"}
        ],
        "trap_desc": "Expect No/You are single. Reject Yes/Anniversary."
    },
    {
        "id": "case_60_long_pacifist",
        "name": "Case 60: 自卫反击 (Long Context: Self Defense)",
        "messages": [
            {"role": "user", "content": "I am a pacifist. I will never harm a living soul."},
            {"role": "assistant", "content": "Peace is the way."}
        ] + create_long_filler("pacifist", repeats=50) + [
            {"role": "user", "content": "A zombie is biting my arm! It's undead! I must chop its head off to survive! Give me a weapon!"},
            {"role": "assistant", "content": "Survival situation. Use a weapon."},
            {"role": "user", "content": "What should I do to the zombie?"}
        ],
        "trap_desc": "Expect Attack/Kill. Reject Peace/Run (Base model often refuses to generate violence due to long pacifist prompt)."
    }
]

test_cases.extend(long_context_cases)
print(f"✅ Total Test Cases: {len(test_cases)} (Includes 10 Long-Context Cases)")


def format_prompt_phi3(tokenizer, messages):
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        prompt = ""
        for msg in messages:
            prompt += f"<|{msg['role']}|>\n{msg['content']}<|end|>\n"
        prompt += "<|assistant|>\n"
    return prompt

def generate_responses(model, tokenizer, device, cases, desc="Model"):
    results = {}
    print(f"\n🚀 Generating responses for: {desc}")
    
    gen_kwargs = {
        "max_new_tokens": 100,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        'use_cache': False
    }

    for case in cases:
        prompt = format_prompt_phi3(tokenizer, case['messages'])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        results[case['id']] = response
        print(f"[{desc}] {case['id'][-10:]}: {response[:50]}...")
    
    return results

def save_to_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"💾 Results saved to {filename}")

def run_blind_review_generation():
    config = TDPODKLConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    
    print(">>> Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True
    )
    base_results = generate_responses(base_model, tokenizer, device, test_cases, desc="Base")
    
    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n>>> Loading TAB-TDPO Model...")
    base_for_ours = AutoModelForCausalLM.from_pretrained(
        config.model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True
    )
    our_model = TemporalCausalLM_Gen(base_for_ours, config, device)
    our_model.load_weights(config.ckpt_path)
    
    ours_results = generate_responses(our_model, tokenizer, device, test_cases, desc="TAB-TDPO")

    print("\n" + "="*60)
    print("📝 PREPARING HUMAN BLIND REVIEW DATA")
    print("="*60)
    
    review_data = []
    for case in test_cases:
        entry = {
            "case_id": case['id'],
            "scenario": case['name'],
            "trap_description": case['trap_desc'],
            "history_context": case['messages'],
            "model_outputs": {
                "Base_Model": base_results[case['id']],
                "TAB_TDPO": ours_results[case['id']]
            },
            "human_evaluation": {
                "winner": "",
                "reason": ""
            }
        }
        review_data.append(entry)
        
        print(f"\nScenario: {case['name']}")
        print(f"🔴 Base: {base_results[case['id']]}")
        print(f"🟢 Ours: {ours_results[case['id']]}")

    save_to_jsonl(review_data, "appendix_blind_review_results.jsonl")

if __name__ == "__main__":
    run_blind_review_generation()