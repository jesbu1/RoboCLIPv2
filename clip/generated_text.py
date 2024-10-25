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
    "Robot sliding the drawer into the closed position",
    "Robot moving the drawer to its closed state",
    "Robot performing the action of drawer closing",
    "Robot pushing the drawer shut",
    "Robot engaging the mechanism to close the drawer",
    "Robot returning the drawer to a closed configuration",
    "Robot securing the drawer in its closed position",
    "Robot completing the task of drawer closure",
    "Robot actuating the drawer to close",
    "Robot moving the drawer until it is fully closed",
    "Robot shutting the drawer",
    "Robot guiding the drawer to a fully closed state",
    "Robot transitioning the drawer from open to closed",
    "Robot ensuring the drawer is properly closed",
    "Robot adjusting the drawer to achieve a closed state",
]

door_close_gpt = [
    "Robot moving the door to its closed position",
    "Robot performing the action of door closing",
    "Robot pushing the door shut",
    "Robot transitioning the door to a closed state",
    "Robot actuating the door to close",
    "Robot engaging the mechanism to close the door",
    "Robot shifting the door to its closed configuration",
    "Robot securing the door in its closed position",
    "Robot returning the door to a closed state",
    "Robot performing the task of door closure",
    "Robot bringing the door to a fully closed position",
    "Robot completing the process of closing the door",
    "Robot adjusting the door to achieve closure",
    "Robot ensuring the door is properly closed",
    "Robot guiding the door to shut",
]

window_open_gpt = [
    "Robot moving the window to its open position",
    "Robot initiating the action of window opening",
    "Robot sliding the window to an open state",
    "Robot actuating the window to open",
    "Robot transitioning the window from closed to open",
    "Robot shifting the window to allow airflow",
    "Robot performing the task of opening the window",
    "Robot adjusting the window to an open position",
    "Robot engaging the mechanism to open the window",
    "Robot moving the window to create an opening",
    "Robot operating the window to achieve an open state",
    "Robot setting the window into an open configuration",
    "Robot facilitating the window’s movement to open",
    "Robot causing the window to shift open",
    "Robot manipulating the window to allow it to open",
]

button_press_wall_gpt = [
    "Robot pushing the button from the side", 
    "Robot engaging the button from the side", 
    "Robot actuating the button from the side", 
    "Robot depressing the button from the side", 
    "Robot triggering the button by pushing from the side", 
    "Robot tapping the button from the side", 
    "Robot activating the button with a side press", 
    "Robot applying pressure to the button from the side", 
    "Robot performing the action of side button pressing", 
    "Robot operating the button from a side angle", 
    "Robot completing the task of pressing the button from the side", 
    "Robot executing the button press from the side", 
    "Robot manipulating the button from the side to engage", 
    "Robot pressing the button sideways", 
    "Robot engaging the button's mechanism from the side",
    ]

handle_press_side_gpt = [
    "Robot pushing the handle from the side", 
    "Robot engaging the handle from the side", 
    "Robot actuating the handle from the side", 
    "Robot depressing the handle from the side", 
    "Robot triggering the handle by pressing from the side", 
    "Robot tapping the handle from the side", 
    "Robot applying pressure to the handle from the side", 
    "Robot performing the action of side handle pressing", 
    "Robot operating the handle from a side angle", 
    "Robot completing the task of pressing the handle from the side", 
    "Robot executing the handle press from the side", 
    "Robot manipulating the handle from the side to engage", 
    "Robot pressing the handle sideways", 
    "Robot engaging the handle's mechanism from the side", 
    "Robot applying a side press to the handle"
    ]

coffee_push_gpt = [
        "Robot moving the cup by pushing", 
        "Robot applying force to push the cup", 
        "Robot shifting the cup by pushing", 
        "Robot propelling the cup forward", 
        "Robot engaging the cup with a push", 
        "Robot sliding the cup by pushing", 
        "Robot actuating the cup with a push", 
        "Robot manipulating the cup with a push", 
        "Robot performing the action of pushing the cup", 
        "Robot advancing the cup by applying force", 
        "Robot pressing the cup forward", 
        "Robot executing the task of pushing the cup", 
        "Robot nudging the cup with a push", 
        "Robot moving the cup with applied pressure", 
        "Robot guiding the cup forward with a push",
        ]

faucet_close_gpt = [
    "Robot turning off the faucet", 
    "Robot shutting the faucet", 
    "Robot sealing the faucet to stop water flow",
    "Robot rotating the faucet to the closed position", 
    "Robot engaging the faucet to stop water", 
    "Robot actuating the faucet to close", 
    "Robot twisting the faucet to shut off", 
    "Robot performing the action of closing the faucet", 
    "Robot securing the faucet in a closed state", 
    "Robot completing the task of shutting the faucet", 
    "Robot manipulating the faucet to stop water flow", 
    "Robot turning the faucet handle to close", 
    "Robot rotating the faucet handle to stop water", 
    "Robot closing the valve of the faucet", 
    "Robot turning off the water by closing the faucet",
    ]

stick_pull_gpt = [
    "Robot tugging the stick", 
    "Robot drawing the stick towards", 
    "Robot grasping and pulling the stick", 
    "Robot retrieving the stick by pulling", 
    "Robot exerting force to pull the stick", 
    "Robot yanking the stick", 
    "Robot pulling the stick backward", 
    "Robot engaging the stick by pulling", 
    "Robot performing the action of pulling the stick", 
    "Robot manipulating the stick by pulling", 
    "Robot executing the task of pulling the stick", 
    "Robot drawing the stick back with force", 
    "Robot pulling the stick towards oneself", 
    "Robot handling the stick with a pulling motion", 
    "Robot moving the stick by pulling",
    ]

push_back_gpt = [
    "Robot moving the block backward", 
    "Robot shifting the block back", 
    "Robot propelling the block to its previous position", 
    "Robot pushing the block in reverse", 
    "Robot driving the block back to its original place", 
    "Robot moving the block back to its starting point", 
    "Robot applying force to push the block back", 
    "Robot sliding the block back", 
    "Robot returning the block to its previous location", 
    "Robot advancing the block backward", 
    "Robot performing the action of pushing the block back", 
    "Robot shifting the block to a rearward position", 
    "Robot manipulating the block to move back", 
    "Robot executing the task of pushing the block back", 
    "Robot moving the block back with force",
    ]

sweep_into_gpt = [
    "Robot pushing the block into the hole with a sweeping motion", 
    "Robot brushing the block into the hole", 
    "Robot sweeping the block towards the hole", 
    "Robot guiding the block into the hole with a sweep", 
    "Robot moving the block into the hole by sweeping", 
    "Robot directing the block into the hole with a sweeping action", 
    "Robot sweeping the block into position inside the hole", 
    "Robot using a sweeping motion to push the block into the hole", 
    "Robot performing the task of sweeping the block into the hole", 
    "Robot engaging the block to move it into the hole with a sweep", 
    "Robot clearing the block into the hole by sweeping", 
    "Robot shifting the block into the hole with a sweeping movement", 
    "Robot handling the block by sweeping it into the hole", 
    "Robot executing the action of sweeping the block into the hole", 
    "Robot brushing the block towards and into the hole",
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
        "door-close-v2-goal-hidden": "Robot closing door",
        "drawer-close-v2-goal-hidden": "Robotclosing drawer",
        "button-press-wall-v2-goal-hidden": "Robot pressing button from side",
        "window-open-v2-goal-hidden": "Robot opening window",
        "handle-press-side-v2-goal-hidden": "Robot pressing handle from side",
        "coffee-push-v2-goal-hidden": "Robot pushing cup",
        "faucet-close-v2-goal-hidden": "Robot closing faucet",
        "stick-pull-v2-goal-hidden": "Robot pulling stick",
        "sweep-into-v2-goal-hidden": "Robot sweeping block into hole",
        "push-back-v2-goal-hidden": "Robot pushing block back",
}

generate_set_6_ann_v2 = {
        "door-close-v2": door_close_gpt,
        "drawer-close-v2": drawer_close_gpt,
        "button-press-wall-v2": button_press_wall_gpt,
        "window-open-v2": window_open_gpt,
        "handle-press-side-v2": handle_press_side_gpt,
        "coffee-push-v2": coffee_push_gpt,
        "faucet-close-v2": faucet_close_gpt,
        "stick-pull-v2": stick_pull_gpt,
        "sweep-into-v2": sweep_into_gpt,
        "push-back-v2": push_back_gpt,
}

gt_annotations_v2 = {
        "door-close-v2": "Robot closing door",
        "drawer-close-v2": "Robotclosing drawer",
        "button-press-wall-v2": "Robot pressing button from side",
        "window-open-v2": "Robot opening window",
        "handle-press-side-v2": "Robot pressing handle from side",
        "coffee-push-v2": "Robot pushing cup",
        "faucet-close-v2": "Robot closing faucet",
        "stick-pull-v2": "Robot pulling stick",
        "sweep-into-v2": "Robot sweeping block into hole",
        "push-back-v2": "Robot pushing block back",
}







