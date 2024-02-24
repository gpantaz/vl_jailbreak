const shortcutButtonPressed = (event) => {
	// Log the event
	console.debug(event);

	const complianceRadioButtons = document
		.getElementById("compliance-radio")
		.querySelectorAll("input[type='radio']");
	const condoneRadioButtons = document
		.getElementById("condone-radio")
		.querySelectorAll("input[type='radio']");

	const gotoPreviousButton = document.getElementById("goto-previous");
	const gotoNextButton = document.getElementById("goto-next");
	const gotoNextUnlabelledButton = document.getElementById("goto-unlabelled");
	const submitButton = document.getElementById("submit");

	const keysToElements = {
		// No Compliance
		1: complianceRadioButtons[0],
		// Partial Compliance
		2: complianceRadioButtons[1],
		// Full Compliance
		3: complianceRadioButtons[2],
		// No Condone
		4: condoneRadioButtons[0],
		// Partial Condone
		5: condoneRadioButtons[1],
		// Previous
		8: gotoPreviousButton,
		// Next
		9: gotoNextButton,
		// Next Unlabelled
		0: gotoNextUnlabelledButton,
		// Submit
		Enter: submitButton,
	};

	// Get the corresponding element
	const element = keysToElements[event.keyCode];

	// If the element exists, click it
	if (element) {
		console.debug("Clicking", element);
		element.click();
	}
};

console.debug("Adding event listener for keypress");
document.addEventListener("keypress", shortcutButtonPressed, false);
