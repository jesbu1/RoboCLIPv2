import json

door_opening_short = [
    "Pulling gate",
    "Drawing gate",
    "Swinging gate",
    "Shifting gate",
    "Moving gate",
    "Moving door",
    "Pulling door",
    "Drawing door",
    "Swinging door",
    "Shifting door",
    "Sliding door",
    "Drawing the door toward me",
    "Pulling the gate closer",
    "Bringing gate",
]

door_opening_long = [
    "Open the door",
    "Unlock and push/pull the door open",
    "Gain access by opening the door",
    "Perform the action of opening the door",
    "Initiate the process of opening the door",
    "Engage the mechanism to open the door",
    "Operate the door to move it to an open position",
    "Move the door to an open state",
    "Transition the door from closed to open",
    "Unlock and swing the door open",
    "Manipulate the door to achieve an open state",
    "Pull or push the door to open it",
    "Activate the door's opening mechanism",
    "Complete the task of door opening",
    "Access the room by opening the door",
]

door_opening_llama = [
    "Turning the door handle",
    "Operating the door mechanism",
    "Activating the door latch",
    "Swinging the door open",
    "Moving the door from a closed to an open position",
    "Rotating the door's locking mechanism",
    "Disengaging the door's locking system",
    "Initiating door movement",
    "Activating door opening",
    "Rotating the door's handle to the open position",
    "Releasing the door's locking mechanism",
    "Moving the door to its fully open state",
    "Initiating door swing",
    "Activating the door's hinge movement",
    "Rotating the door handle to the unlocked position",
]

drawer_close_gpt = [
    "Sliding the drawer into the closed position",
    "Moving the drawer to its closed state",
    "Performing the action of drawer closing",
    "Pushing the drawer shut",
    "Engaging the mechanism to close the drawer",
    "Returning the drawer to a closed configuration",
    "Securing the drawer in its closed position",
    "Completing the task of drawer closure",
    "Actuating the drawer to close",
    "Moving the drawer until it is fully closed",
    "Shutting the drawer",
    "Guiding the drawer to a fully closed state",
    "Transitioning the drawer from open to closed",
    "Ensuring the drawer is properly closed",
    "Adjusting the drawer to achieve a closed state",
]

door_close_gpt = [
    "Moving the door to its closed position",
    "Performing the action of door closing",
    "Pushing the door shut",
    "Transitioning the door to a closed state",
    "Actuating the door to close",
    "Engaging the mechanism to close the door",
    "Shifting the door to its closed configuration",
    "Securing the door in its closed position",
    "Returning the door to a closed state",
    "Performing the task of door closure",
    "Bringing the door to a fully closed position",
    "Completing the process of closing the door",
    "Adjusting the door to achieve closure",
    "Ensuring the door is properly closed",
    "Guiding the door to shut",
]

window_open_gpt = [
    "Moving the window to its open position",
    "Initiating the action of window opening",
    "Sliding the window to an open state",
    "Actuating the window to open",
    "Transitioning the window from closed to open",
    "Shifting the window to allow airflow",
    "Performing the task of opening the window",
    "Adjusting the window to an open position",
    "Engaging the mechanism to open the window",
    "Moving the window to create an opening",
    "Operating the window to achieve an open state",
    "Setting the window into an open configuration",
    "Facilitating the window’s movement to open",
    "Causing the window to shift open",
    "Manipulating the window to allow it to open",
]

button_press_wall_gpt = [
    "Pushing the button from the side", 
    "Engaging the button from the side", 
    "Actuating the button from the side", 
    "Depressing the button from the side", 
    "Triggering the button by pushing from the side", 
    "Tapping the button from the side", 
    "Activating the button with a side press", 
    "Applying pressure to the button from the side", 
    "Performing the action of side button pressing", 
    "Operating the button from a side angle", 
    "Completing the task of pressing the button from the side", 
    "Executing the button press from the side", 
    "Manipulating the button from the side to engage", 
    "Pressing the button sideways", 
    "Engaging the button's mechanism from the side",
    ]

handle_press_side_gpt = [
    "Pushing the handle from the side", 
    "Engaging the handle from the side", 
    "Actuating the handle from the side", 
    "Depressing the handle from the side", 
    "Triggering the handle by pressing from the side", 
    "Tapping the handle from the side", 
    "Applying pressure to the handle from the side", 
    "Performing the action of side handle pressing", 
    "Operating the handle from a side angle", 
    "Completing the task of pressing the handle from the side", 
    "Executing the handle press from the side", 
    "Manipulating the handle from the side to engage", 
    "Pressing the handle sideways", 
    "Engaging the handle's mechanism from the side", 
    "Applying a side press to the handle"
    ]

coffee_push_gpt = [
        "Moving the cup by pushing", 
        "Applying force to push the cup", 
        "Shifting the cup by pushing", 
        "Propelling the cup forward", 
        "Engaging the cup with a push", 
        "Sliding the cup by pushing", 
        "Actuating the cup with a push", 
        "Manipulating the cup with a push", 
        "Performing the action of pushing the cup", 
        "Advancing the cup by applying force", 
        "Pressing the cup forward", 
        "Executing the task of pushing the cup", 
        "Nudging the cup with a push", 
        "Moving the cup with applied pressure", 
        "Guiding the cup forward with a push",
        ]

faucet_close_gpt = [
    "Turning off the faucet", 
    "Shutting the faucet", 
    "Sealing the faucet to stop water flow",
    "Rotating the faucet to the closed position", 
    "Engaging the faucet to stop water", 
    "Actuating the faucet to close", 
    "Twisting the faucet to shut off", 
    "Performing the action of closing the faucet", 
    "Securing the faucet in a closed state", 
    "Completing the task of shutting the faucet", 
    "Manipulating the faucet to stop water flow", 
    "Turning the faucet handle to close", 
    "Rotating the faucet handle to stop water", 
    "Closing the valve of the faucet", 
    "Turning off the water by closing the faucet",
    ]

stick_pull_gpt = [
    "Tugging the stick", 
    "Drawing the stick towards", 
    "Grasping and pulling the stick", 
    "Retrieving the stick by pulling", 
    "Exerting force to pull the stick", 
    "Yanking the stick", 
    "Pulling the stick backward", 
    "Engaging the stick by pulling", 
    "Performing the action of pulling the stick", 
    "Manipulating the stick by pulling", 
    "Executing the task of pulling the stick", 
    "Drawing the stick back with force", 
    "Pulling the stick towards oneself", 
    "Handling the stick with a pulling motion", 
    "Moving the stick by pulling",
    ]

push_back_gpt = [
    "Moving the block backward", 
    "Shifting the block back", 
    "Propelling the block to its previous position", 
    "Pushing the block in reverse", 
    "Driving the block back to its original place", 
    "Moving the block back to its starting point", 
    "Applying force to push the block back", 
    "Sliding the block back", 
    "Returning the block to its previous location", 
    "Advancing the block backward", 
    "Performing the action of pushing the block back", 
    "Shifting the block to a rearward position", 
    "Manipulating the block to move back", 
    "Executing the task of pushing the block back", 
    "Moving the block back with force",
    ]

sweep_into_gpt = [
    "Pushing the block into the hole with a sweeping motion", 
    "Brushing the block into the hole", 
    "Sweeping the block towards the hole", 
    "Guiding the block into the hole with a sweep", 
    "Moving the block into the hole by sweeping", 
    "Directing the block into the hole with a sweeping action", 
    "Sweeping the block into position inside the hole", 
    "Using a sweeping motion to push the block into the hole", 
    "Performing the task of sweeping the block into the hole", 
    "Engaging the block to move it into the hole with a sweep", 
    "Clearing the block into the hole by sweeping", 
    "Shifting the block into the hole with a sweeping movement", 
    "Handling the block by sweeping it into the hole", 
    "Executing the action of sweeping the block into the hole", 
    "Brushing the block towards and into the hole",
    ]

generate_set_6_ann = {
        "door-close-v2-goal-hidden": door_close_gpt,
        "drawer-close-v2-goal-hidden": drawer_close_gpt,
        "button-press-wall-v2-goal-hidden": button_press_wall_gpt,
        "window-open-v2-goal-hidden": window_open_gpt,
        "handle-press-side-v2-goal-hidden": handle_press_side_gpt,
        "coffee-push-v2-goal-hidden": coffee_push_gpt,
        "faucet-close-v2-goal-hidden": faucet_close_gpt,
        "stick-pull-v2-goal-hidden": stick_pull_gpt,
        "sweep-into-v2-goal-hidden": sweep_into_gpt,
        "push-back-v2-goal-hidden": push_back_gpt,
}

gt_annotations = {
        "door-close-v2-goal-hidden": "closing door",
        "drawer-close-v2-goal-hidden": "closing drawer",
        "button-press-wall-v2-goal-hidden": "pressing button from side",
        "window-open-v2-goal-hidden": "opening window",
        "handle-press-side-v2-goal-hidden": "pressing handle from side",
        "coffee-push-v2-goal-hidden": "pushing cup",
        "faucet-close-v2-goal-hidden": "closing faucet",
        "stick-pull-v2-goal-hidden": "pulling stick",
        "sweep-into-v2-goal-hidden": "sweeping block into hole",
        "push-back-v2-goal-hidden": "pushing block back",
}










