const fs = require('fs');

function datasetToArray(dataset) {
  const lines = dataset.trim().split('\n');
  const items = [];
  let currentItem = [];

  for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();

      if (line !== '') {
          currentItem.push(line);
      }

      // Check if this is the end of a conversation turn
      if (currentItem.length === 3) {
          items.push(currentItem);
          currentItem = [];
      }
  }

  return items;
}

// Example usage
const dataset = fs.readFileSync('./emojis.txt', 'utf8');

const datasetArray = datasetToArray(dataset);

function processAssistantResponse(response) {
  // Extract text after "Assistant:"
  const textAfterAssistant = response.split('Assistant:')[1].trim();

  // Split the text into an array of words
  const words = textAfterAssistant.split(' ');

  // Replace emojis with U+... representation
  const processedWords = words.map(word => {
      // Check if the word is an emoji
      const emojiMatch = word.match(/\p{Emoji}/gu);
      if (emojiMatch) {
          // Replace the emoji with its U+... representation
          return emojiMatch.map(emoji => `U+${emoji.codePointAt(0).toString(16).toUpperCase()}`).join('');
      } else {
          return word;
      }
  });

  // Join the processed words back into a string
  const processedResponse = processedWords.join(' ');

  return processedResponse;
}

const processedDataset = datasetArray.map(item => {
  const processedResponse = processAssistantResponse(item[2]);
  return [item[0], item[1], `Assistant: ${processedResponse}`];
});

const string = processedDataset.map(item => item.join('\n')).join('\n\n')

fs.writeFileSync('./unicode_emojis.txt', string, 'utf8');