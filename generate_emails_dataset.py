import json

data = [
("Send me the report today. You missed it yesterday.", "Hey, could you please send the report today? Just wanted to check in since we missed it yesterday."),
("I hate this design. Do it again.", "I think we can improve this design a bit. Want to try a few more options?"),
("Fix this bug immediately.", "Hey, could you please take a look at this bug when you get a chance? It would really help to get it resolved soon."),
("I'm not doing this task. It's stupid.", "I am not fully convinced about this task. Can we talk it through?"),
("Give me the slides by noon.", "Could you please share the slides by noon? Thanks!"),
("Your code is garbage and broke the build.", "Looks like the recent code may have impacted the build. Can we review it together?"),
("Stop emailing me about this.", "Hey, would it be okay if we pause the email updates on this for now?"),
("The client is super annoying.", "The client has been a bit challenging lately."),
("Why did you do it this way? It makes no sense.", "Hey, could you walk me through your approach here? I would love to understand it better."),
("Tell John he is wrong about the budget.", "Could you check with John? I think there might be a small mismatch in the budget numbers."),
("This meeting was a waste of time.", "I think we could make our meetings more productive going forward."),
("You completely ignored my email.", "Hey, just checking. Did you get a chance to look at my earlier email?"),
("I don't care what they say, we are doing it my way.", "Let us consider their feedback, but I still feel this approach could work well."),
("You're late again.", "Hey, I noticed you have been running a bit late. Everything okay?"),
("This is the worst idea I've ever heard.", "I have some concerns about this idea. Maybe we can explore a few alternatives?"),
("Fix the spelling mistake right now.", "Could you please fix the spelling typo when you get a moment?"),
("I am too busy for this nonsense.", "I am a bit tied up right now, so I might not be able to take this on."),
("You clearly didn't read my instructions.", "It seems there might have been a small misunderstanding. Happy to clarify if needed."),
("Do not interrupt me when I am speaking.", "Hey, could you let me finish my thought? Thanks!"),
("You messed up the presentation.", "There are a few areas in the presentation we can improve. Let us refine them together."),
("This code is a disaster.", "This code might benefit from a bit of cleanup and refactoring."),
("I expect this done by tomorrow, no excuses.", "It would be great if we could have this ready by tomorrow."),
("Your performance has been terrible.", "Let us connect and talk about a few areas where we can improve."),
("I'm done dealing with this client.", "I think it might help if someone else steps in to handle this client for now."),
("Why is this taking so long?", "Hey, could you share an update on the timeline?"),
("You need to learn how to communicate.", "I think clearer communication could really help our teamwork."),
("Don't ask me for help again.", "I am a bit stretched right now, so I may not be able to help further."),
("I refuse to work with him.", "I am finding it a bit challenging to collaborate. Can we explore other options?"),
("Your report is full of lies.", "I noticed a few inconsistencies in the report. Can we review them together?"),
("This project is a complete joke.", "I have some concerns about how this project is shaping up."),
("You need to stop being so lazy.", "I would really appreciate seeing more consistency in effort on these tasks."),
("I don't want to hear your excuses.", "Let us focus on how we can move forward and solve this."),
("I told you so.", "This seems to align with something I mentioned earlier."),
("You're not paid to think.", "Let us stick to the current approach for now."),
("This is not my problem.", "I think this might be outside my current scope."),
("You are wrong.", "I see this a bit differently. Happy to discuss."),
("I'm muting this chat.", "I am going to mute this chat for now so I can stay focused."),
("Do whatever you want, I don't care.", "I am okay with you taking the lead on this."),
("You clearly have no idea what you're doing.", "It might help to go over this together or get a bit more context."),
("This is a hard no.", "I am afraid I will not be able to approve this."),
("I'm logging off early because this is useless.", "I am going to log off for now since I am a bit blocked here."),
("Learn how to use Git properly.", "Maybe we can go over some Git best practices together?"),
("This is above your pay grade.", "This might need input from leadership."),
("I already answered this. Read the thread.", "I have shared some details earlier in the thread. Feel free to take a look.")
]
out_filename = "emails.jsonl"
with open(out_filename, "w") as f:
    for blunt, prof in data:
        record = {
            "instruction": f"Rewrite friendly: {blunt}",
            "output": prof
        }
        f.write(json.dumps(record) + "\n")

print(f"Generated {len(data)} examples in {out_filename}")
