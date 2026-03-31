export interface Character {
  id: string;
  name: string;
  aliases: string[];
  mention_count: number;
}

export interface Mention {
  source: string;
  target: string;
  context: string;
  chapter?: string;
  date?: string;
}

export interface GraphData {
  characters: Character[];
  mentions: Mention[];
}
