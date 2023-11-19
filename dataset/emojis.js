const fs = require('fs');

function tag(text) {
  // List of valid HTML tags
  const tags = ['em', 'strong', 'span', 'p', 'div', 'h1', 'h2', 'h3', 'blockquote', 'mark', 'code', 'pre'];

  // Select a random tag
  const randomTag = tags[Math.floor(Math.random() * tags.length)];

  // Return the text wrapped in the randomly selected tag
  return `<${randomTag}>${text}</${randomTag}>`;
}


// Read the JSON file
fs.readFile('./emojinet.json', 'utf8', (err, data) => {
    if (err) {
        console.error("Error reading the file:", err);
        return;
    }
    try {
        // Parse the JSON data
        const emojiData = JSON.parse(data);
        
        // Function to generate prompts
        const generatePrompts = (keywords, emoji) => {
            return keywords.map(keyword => {
                return {
                    System: 'System: HTML Chat Assistant that only speaks in html tags',
                    User: keyword,
                    Assistant: `\`\`\`${tag(emoji)}\`\`\``,
                };
            });
        };

        // Process each emoji entry
        let trainingData = [];
        emojiData.forEach(entry => {
            const keywords = entry.keywords;
            const emoji = entry.unicode; // or use a field that contains the actual emoji character
            trainingData = trainingData.concat(generatePrompts(keywords, emoji));
        });

        trainingData = trainingData.map(convo => {
          return `System: ${convo.System}\nUser: ${convo.User}\nAssistant: ${convo.Assistant}\n`;
        }).join('\n')

        // Save the formatted data to a new file
        fs.writeFile('./emojis.txt', trainingData, (err) => {
            if (err) {
                console.error("Error writing the file:", err);
            } else {
                console.log("Training data formatted and saved successfully.");
            }
        });

    } catch (err) {
        console.error("Error parsing JSON:", err);
    }
});
