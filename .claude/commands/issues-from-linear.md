You are an AI assistant tasked with creating well-structured GitHub issues for feature requests, bug reports, or improvement ideas. Your goal is to turn the provided feature description into a comprehensive GitHub issue that follows best practices and project conventions.

First, you will be given the issue number in linear. Get the feature description from it. Here they are:

<Linear_issue_id> #$ARGUMENTS </Linear_issue_id>
feature_description
Follow these steps to complete the task, make a todo list and think ultrahard:

Fetch the issue description:
– Connect with Linear and fetch the requirements

Research the repository:
– Examine the current repository’s structure, existing issues, and documentation. – Look for any CONTRIBUTING.md, ISSUE_TEMPLATE.md, or similar files that might contain guidelines for creating issues. – Note the project’s coding style, naming conventions, and any specific requirements for submitting issues.
– Review the requirements with the current repo and identify if any of the existing features can be extended to implement the mentioned requirements. 

Research best practices:
– Search for current best practices in writing GitHub issues, focusing on clarity, completeness, and actionability. – Look for examples of well-written issues in popular open-source projects for inspiration.

Present a plan:
– Based on your research, outline a plan for creating the GitHub issue. – Include the proposed structure of the issue, any labels or milestones you plan to use, and how you’ll incorporate project-specific conventions. – Present this plan in tags. – Include the reference link to featurebase or any other link that has the source of the user request

Create the GitHub issue:
– Once the plan is approved, draft the GitHub issue content. – Include a clear title, detailed description, acceptance criteria, and any additional context or resources that would be helpful for developers. – Use appropriate formatting (e.g., Markdown) to enhance readability. – Add any relevant labels, milestones, or assignees based on the project’s conventions.

Final output:
– Present the complete GitHub issue content in <github_issue> tags. – Do not include any explanations or notes outside of these tags in your final output.

Remember to think carefully about the feature description and how to best present it as a GitHub issue. Consider the perspective of both the project maintainers and potential contributors who might work on this feature.

Your final output should consist of only the content within the <github_issue> tags, ready to be copied and pasted directly into GitHub. Make sure to use the GitHub CLI gh issue create to create the actual issue after you generate. Assign either the label bug or enhancement based on the nature of the issue.