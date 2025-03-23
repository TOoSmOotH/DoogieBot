interface ThinkPart {
  content: string;
  isThink: boolean;
  isComplete: boolean;
}

/**
 * Parse content with think tags into an array of parts
 * Each part is either regular content or think tag content
 *
 * @param content The content to parse
 * @returns Array of parsed parts
 */
export const parseThinkTags = (content: string): ThinkPart[] => {
  const parts: ThinkPart[] = [];
  const thinkRegex = /<think>([\s\S]*?)<\/think>/g;
  const incompleteThinkRegex = /<think>([\s\S]*)$/;
  let lastIndex = 0;

  // Reset regex lastIndex
  thinkRegex.lastIndex = 0;

  // Extract complete think tags
  let match;
  while ((match = thinkRegex.exec(content)) !== null) {
    // Add text before the think tag
    if (match.index > lastIndex) {
      parts.push({
        content: content.slice(lastIndex, match.index),
        isThink: false,
        isComplete: true,
      });
    }

    // Add think tag content
    parts.push({
      content: match[1],
      isThink: true,
      isComplete: true,
    });

    lastIndex = match.index + match[0].length;
  }

  // Check for incomplete think tag
  const incompleteMatch = content.match(incompleteThinkRegex);
  if (
    incompleteMatch &&
    incompleteMatch.index !== undefined &&
    incompleteMatch.index >= lastIndex &&
    !content.includes('</think>', incompleteMatch.index)
  ) {
    // Add text before the incomplete think tag
    if (incompleteMatch.index > lastIndex) {
      parts.push({
        content: content.slice(lastIndex, incompleteMatch.index),
        isThink: false,
        isComplete: true,
      });
    }

    // Add incomplete think tag content
    parts.push({
      content: incompleteMatch[1],
      isThink: true,
      isComplete: false,
    });

    lastIndex = content.length;
  }

  // Add remaining text after last think tag
  if (lastIndex < content.length) {
    parts.push({
      content: content.slice(lastIndex),
      isThink: false,
      isComplete: true,
    });
  }

  return parts;
};
