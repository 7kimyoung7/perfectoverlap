
import React, { useState, useEffect, useRef, useMemo } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI } from "@google/genai";

// --- Types ---
type Coordinate = { x: number; y: number };
type SubSquare = { x: number; y: number; size: number };
type Piece = {
  id: number;
  color: string;
  cells: Coordinate[]; // Relative coordinates for collision logic
  squares: SubSquare[]; // Visual representation: decomposed into squares
  position: Coordinate | null; // null = in inventory, otherwise board coordinates
  solution: Coordinate; // The correct position for this piece (minX, minY)
  width: number;
  height: number;
};

// --- Constants ---
const GRID_SIZE = 10;
const NUM_PIECES = 10;
// Reduced cell size to 34px to fit everything in one page better
const CELL_PIXEL_SIZE = 34; 

// --- Helper Functions ---

// Decompose a set of grid cells into the largest possible squares (Greedy approach)
const decomposeToSquares = (cells: Coordinate[]): SubSquare[] => {
  if (cells.length === 0) return [];

  // Determine bounds
  const xs = cells.map(c => c.x);
  const ys = cells.map(c => c.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const width = maxX - minX + 1;
  const height = maxY - minY + 1;

  // Create a boolean map
  const map = Array(height).fill(false).map(() => Array(width).fill(false));
  cells.forEach(c => {
    map[c.y - minY][c.x - minX] = true;
  });

  const squares: SubSquare[] = [];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (map[y][x]) {
        // Found a top-left corner of a potential square
        // Find max size
        let maxSize = 1;
        // Check progressively larger squares
        while (true) {
          const nextSize = maxSize + 1;
          if (y + nextSize > height || x + nextSize > width) break;
          
          let clean = true;
          // Check the new row and column added by expanding
          for (let i = 0; i < nextSize; i++) {
             if (!map[y + nextSize - 1][x + i] || !map[y + i][x + nextSize - 1]) {
               clean = false;
               break;
             }
          }
          
          if (clean) {
            maxSize = nextSize;
          } else {
            break;
          }
        }

        // Record square
        squares.push({ x: x + minX, y: y + minY, size: maxSize });

        // Mark covered cells as used (false)
        for (let dy = 0; dy < maxSize; dy++) {
          for (let dx = 0; dx < maxSize; dx++) {
            map[y + dy][x + dx] = false;
          }
        }
      }
    }
  }
  return squares;
};

// Ensure all pieces are contiguous.
const ensureContiguity = (grid: number[][]) => {
    let changed = true;
    while (changed) {
        changed = false;
        // For each piece ID
        for (let id = 0; id < NUM_PIECES; id++) {
            // Find all cells
            const cells: Coordinate[] = [];
            for(let y=0; y<GRID_SIZE; y++) {
                for(let x=0; x<GRID_SIZE; x++) {
                    if (grid[y][x] === id) cells.push({x, y});
                }
            }
            if (cells.length === 0) continue;

            // BFS to find connected components
            const visited = new Set<string>();
            const components: Coordinate[][] = [];
            
            for (const cell of cells) {
                const key = `${cell.x},${cell.y}`;
                if (visited.has(key)) continue;

                const component: Coordinate[] = [];
                const queue = [cell];
                visited.add(key);
                
                while (queue.length > 0) {
                    const curr = queue.shift()!;
                    component.push(curr);
                    const dirs = [[0, 1], [0, -1], [1, 0], [-1, 0]];
                    for (const [dy, dx] of dirs) {
                        const nx = curr.x + dx;
                        const ny = curr.y + dy;
                        const nKey = `${nx},${ny}`;
                        if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE && 
                            grid[ny][nx] === id && !visited.has(nKey)) {
                            visited.add(nKey);
                            queue.push({x: nx, y: ny});
                        }
                    }
                }
                components.push(component);
            }

            // If more than 1 component, merge smaller ones into neighbors
            if (components.length > 1) {
                // Sort by size descending
                components.sort((a, b) => b.length - a.length);
                
                // Keep the largest component (index 0), reassign others
                for (let i = 1; i < components.length; i++) {
                    const toReassign = components[i];
                    for (const cell of toReassign) {
                        // Find a neighbor that is NOT this piece
                        const dirs = [[0, 1], [0, -1], [1, 0], [-1, 0]];
                        let newId = -1;
                        for (const [dy, dx] of dirs) {
                            const nx = cell.x + dx;
                            const ny = cell.y + dy;
                            if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE && grid[ny][nx] !== id) {
                                newId = grid[ny][nx];
                                break;
                            }
                        }
                        // Assign to neighbor or fallback
                        if (newId !== -1) {
                            grid[cell.y][cell.x] = newId;
                            changed = true;
                        }
                    }
                }
            }
        }
    }
};

const generatePuzzle = (): Piece[] => {
  const grid = Array(GRID_SIZE).fill(null).map(() => Array(GRID_SIZE).fill(-1));
  
  // Initialize seeds
  let seeds: Coordinate[] = [];
  while (seeds.length < NUM_PIECES) {
    const x = Math.floor(Math.random() * GRID_SIZE);
    const y = Math.floor(Math.random() * GRID_SIZE);
    if (grid[y][x] === -1) {
      const tooClose = seeds.some(s => Math.abs(s.x - x) + Math.abs(s.y - y) < 2);
      if (!tooClose || seeds.length > 8) {
         grid[y][x] = seeds.length;
         seeds.push({ x, y });
      }
    }
  }

  // Grow regions
  let changed = true;
  while (changed) {
    changed = false;
    const pieceOrder = Array.from({ length: NUM_PIECES }, (_, i) => i).sort(() => Math.random() - 0.5);
    
    for (const pieceId of pieceOrder) {
      const candidates: { x: number, y: number, score: number }[] = [];
      for (let y = 0; y < GRID_SIZE; y++) {
        for (let x = 0; x < GRID_SIZE; x++) {
          if (grid[y][x] === -1) {
             const dirs = [[0, 1], [0, -1], [1, 0], [-1, 0]];
             let neighborsOfSameId = 0;
             let hasNeighbor = false;
             for (const [dy, dx] of dirs) {
                const ny = y + dy;
                const nx = x + dx;
                if (ny >= 0 && ny < GRID_SIZE && nx >= 0 && nx < GRID_SIZE) {
                   if (grid[ny][nx] === pieceId) {
                      hasNeighbor = true;
                      neighborsOfSameId++;
                   }
                }
             }
             if (hasNeighbor) {
                candidates.push({ x, y, score: neighborsOfSameId });
             }
          }
        }
      }
      if (candidates.length > 0) {
        candidates.sort((a, b) => b.score - a.score || Math.random() - 0.5);
        const pick = candidates[0];
        grid[pick.y][pick.x] = pieceId;
        changed = true;
      }
    }
  }

  // Final Cleanup
  for (let y = 0; y < GRID_SIZE; y++) {
    for (let x = 0; x < GRID_SIZE; x++) {
      if (grid[y][x] === -1) {
        const neighbors = [[0, 1], [0, -1], [1, 0], [-1, 0]];
        for (const [dy, dx] of neighbors) {
           const ny = y + dy;
           const nx = x + dx;
           if (ny >= 0 && ny < GRID_SIZE && nx >= 0 && nx < GRID_SIZE && grid[ny][nx] !== -1) {
             grid[y][x] = grid[ny][nx];
             break;
           }
        }
        if (grid[y][x] === -1) grid[y][x] = 0;
      }
    }
  }

  ensureContiguity(grid);

  // Convert to Piece objects
  const pieces: Piece[] = [];
  const colors = [
    '#F59E0B', '#D97706', '#EA580C', '#CA8A04', '#EAB308', 
    '#F97316', '#FB923C', '#FBBF24', '#FCD34D', '#fdba74'
  ];

  for (let id = 0; id < NUM_PIECES; id++) {
    const rawCells: Coordinate[] = [];
    let minX = GRID_SIZE, maxX = 0, minY = GRID_SIZE, maxY = 0;

    for (let y = 0; y < GRID_SIZE; y++) {
      for (let x = 0; x < GRID_SIZE; x++) {
        if (grid[y][x] === id) {
          rawCells.push({ x, y });
          minX = Math.min(minX, x);
          maxX = Math.max(maxX, x);
          minY = Math.min(minY, y);
          maxY = Math.max(maxY, y);
        }
      }
    }

    const normalizedCells = rawCells.map(c => ({ x: c.x - minX, y: c.y - minY }));
    const squares = decomposeToSquares(normalizedCells);

    pieces.push({
      id,
      color: colors[id % colors.length],
      cells: normalizedCells,
      squares: squares,
      position: null,
      solution: { x: minX, y: minY }, // Store solution
      width: maxX - minX + 1,
      height: maxY - minY + 1
    });
  }

  return pieces;
};

// --- Components ---

const Game = () => {
  const [pieces, setPieces] = useState<Piece[]>([]);
  const [isClient, setIsClient] = useState(false);
  const [draggingPieceId, setDraggingPieceId] = useState<number | null>(null);
  const [dragOffset, setDragOffset] = useState<Coordinate>({ x: 0, y: 0 });
  const [ghostPos, setGhostPos] = useState<Coordinate | null>(null);
  const boardRef = useRef<HTMLDivElement>(null);
  
  // Modal State
  const [modal, setModal] = useState<{ show: boolean; title: string; message: string; type: 'success' | 'error' | 'info' } | null>(null);

  // Timer State
  const [timeElapsed, setTimeElapsed] = useState(0);
  const [isGameActive, setIsGameActive] = useState(false);

  const startNewGame = () => {
    setPieces(generatePuzzle());
    setTimeElapsed(0);
    setIsGameActive(true);
    setModal(null);
  };

  useEffect(() => {
    startNewGame();
    setIsClient(true);
  }, []);

  // Timer Logic
  useEffect(() => {
    let interval: any;
    if (isGameActive) {
      interval = setInterval(() => {
        setTimeElapsed(prev => prev + 1);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isGameActive]);

  // Automatic Victory Detection
  useEffect(() => {
    if (!isGameActive || pieces.length === 0) return;

    // Check if all pieces are placed
    const allPlaced = pieces.every(p => p.position !== null);
    if (!allPlaced) return;

    // Check for perfect coverage
    const grid = new Set<string>();
    let overlap = false;
    for (const p of pieces) {
      if (!p.position) continue;
      for (const cell of p.cells) {
        const key = `${p.position.x + cell.x},${p.position.y + cell.y}`;
        if (grid.has(key)) {
            overlap = true;
            break;
        }
        grid.add(key);
      }
      if (overlap) break;
    }

    if (!overlap && grid.size === GRID_SIZE * GRID_SIZE) {
      setIsGameActive(false);
      setModal({
        show: true,
        title: "挑战成功",
        message: "红方选手张天扬守擂成功\n【通关密钥】为301",
        type: 'success'
      });
    }
  }, [pieces, isGameActive]);

  // --- Handlers ---

  const handleDragStart = (e: React.DragEvent, piece: Piece, offsetX: number, offsetY: number) => {
    setDraggingPieceId(piece.id);
    setDragOffset({ x: offsetX, y: offsetY });
    e.dataTransfer.effectAllowed = "move";
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";

    if (draggingPieceId === null || !boardRef.current) return;

    // 1. Calculate grid coordinates from mouse position
    const rect = boardRef.current.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / CELL_PIXEL_SIZE);
    const y = Math.floor((e.clientY - rect.top) / CELL_PIXEL_SIZE);

    // 2. Adjust for the grab offset
    const targetX = x - dragOffset.x;
    const targetY = y - dragOffset.y;

    // 3. Optimization: Don't re-validate if position hasn't changed
    if (ghostPos && ghostPos.x === targetX && ghostPos.y === targetY) return;

    // 4. Validate Placement for Ghost
    const piece = pieces.find(p => p.id === draggingPieceId);
    if (!piece) return;

    let isValid = true;
    
    // Bounds check
    for (const cell of piece.cells) {
        const absX = targetX + cell.x;
        const absY = targetY + cell.y;
        if (absX < 0 || absX >= GRID_SIZE || absY < 0 || absY >= GRID_SIZE) {
            isValid = false;
            break;
        }
    }

    // Overlap check
    if (isValid) {
        for (const other of pieces) {
            if (other.id === draggingPieceId || !other.position) continue;
            for (const myCell of piece.cells) {
                const myAbsX = targetX + myCell.x;
                const myAbsY = targetY + myCell.y;
                for (const otherCell of other.cells) {
                    const otherAbsX = other.position.x + otherCell.x;
                    const otherAbsY = other.position.y + otherCell.y;
                    if (myAbsX === otherAbsX && myAbsY === otherAbsY) {
                        isValid = false;
                        break;
                    }
                }
                if (!isValid) break;
            }
            if (!isValid) break;
        }
    }

    if (isValid) {
        setGhostPos({ x: targetX, y: targetY });
    } else {
        setGhostPos(null);
    }
  };

  const handleDragLeave = (e: React.DragEvent) => {
    // Optional: Clear ghost if leaving board completely
  };

  const handleDropOnGrid = (e: React.DragEvent) => {
    e.preventDefault();
    setGhostPos(null); // Clear ghost

    if (draggingPieceId === null) return;
    
    if (!boardRef.current) return;
    const rect = boardRef.current.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / CELL_PIXEL_SIZE);
    const y = Math.floor((e.clientY - rect.top) / CELL_PIXEL_SIZE);
    const newX = x - dragOffset.x;
    const newY = y - dragOffset.y;

    const piece = pieces.find(p => p.id === draggingPieceId);
    if (!piece) return;

    // Re-validate (Drop logic)
    // 1. Bounds
    for (const cell of piece.cells) {
      const absX = newX + cell.x;
      const absY = newY + cell.y;
      if (absX < 0 || absX >= GRID_SIZE || absY < 0 || absY >= GRID_SIZE) return;
    }
    // 2. Overlap
    for (const other of pieces) {
      if (other.id === draggingPieceId || !other.position) continue;
      for (const myCell of piece.cells) {
        const myAbsX = newX + myCell.x;
        const myAbsY = newY + myCell.y;
        for (const otherCell of other.cells) {
          const otherAbsX = other.position.x + otherCell.x;
          const otherAbsY = other.position.y + otherCell.y;
          if (myAbsX === otherAbsX && myAbsY === otherAbsY) return;
        }
      }
    }

    setPieces(prev => prev.map(p => 
      p.id === draggingPieceId ? { ...p, position: { x: newX, y: newY } } : p
    ));
    setDraggingPieceId(null);
  };

  const handleDropOnInventory = (e: React.DragEvent) => {
    e.preventDefault();
    setGhostPos(null);
    if (draggingPieceId === null) return;
    
    setPieces(prev => prev.map(p => 
      p.id === draggingPieceId ? { ...p, position: null } : p
    ));
    setDraggingPieceId(null);
  };

  // --- Hint System ---
  const handleHint = () => {
    if (!isGameActive) return; // Prevent hint if game finished

    const unplacedPieces = pieces.filter(p => p.position === null);
    if (unplacedPieces.length === 0) return;

    // Pick a random unplaced piece
    const randomPiece = unplacedPieces[Math.floor(Math.random() * unplacedPieces.length)];
    
    setPieces(prev => {
        const newPieces = [...prev];
        const pieceToPlace = newPieces.find(p => p.id === randomPiece.id)!;
        
        // Find if any existing placed piece clashes with the solution
        const solution = pieceToPlace.solution;
        const piecesToKick: number[] = [];

        // Check collision against all currently placed pieces
        newPieces.forEach(other => {
            if (other.id === pieceToPlace.id || other.position === null) return;
            
            // Check intersection
            let intersects = false;
            for (const myCell of pieceToPlace.cells) {
                const myAbsX = solution.x + myCell.x;
                const myAbsY = solution.y + myCell.y;
                for (const otherCell of other.cells) {
                    const otherAbsX = other.position!.x + otherCell.x;
                    const otherAbsY = other.position!.y + otherCell.y;
                    if (myAbsX === otherAbsX && myAbsY === otherAbsY) {
                        intersects = true;
                        break;
                    }
                }
                if (intersects) break;
            }
            if (intersects) piecesToKick.push(other.id);
        });

        // Kick intersected pieces back to inventory
        piecesToKick.forEach(pid => {
            const p = newPieces.find(x => x.id === pid)!;
            p.position = null;
        });

        // Place the hint piece
        pieceToPlace.position = solution;
        
        return newPieces;
    });
  };

  // Manual Submit
  const handleSubmit = () => {
    // Check unplaced
    const unplaced = pieces.filter(p => p.position === null).length;
    if (unplaced > 0) {
        setModal({
            show: true,
            title: "挑战未完成",
            message: `还有 ${unplaced} 个碎片未放置！`,
            type: 'info'
        });
        return;
    }

    // Check overlap and coverage
    const grid = new Set<string>();
    let overlap = false;
    for (const p of pieces) {
        if (!p.position) continue;
        for (const cell of p.cells) {
            const key = `${p.position.x + cell.x},${p.position.y + cell.y}`;
            if (grid.has(key)) overlap = true;
            grid.add(key);
        }
    }

    if (!overlap && grid.size === GRID_SIZE * GRID_SIZE) {
        setIsGameActive(false);
        setModal({
            show: true,
            title: "挑战成功",
            message: "红方选手张天扬守擂成功！\n【通关密钥】为301",
            type: 'success'
        });
    } else {
        setModal({
            show: true,
            title: "判定失败",
            message: "碎片存在重叠或未正确填满区域，请检查。",
            type: 'error'
        });
    }
  };

  const renderPiece = (piece: Piece, isInventory: boolean, isGhost: boolean = false) => {
    const scale = 1; // Always 1:1 scale for real proportion
    const width = piece.width * CELL_PIXEL_SIZE * scale;
    const height = piece.height * CELL_PIXEL_SIZE * scale;

    return (
      <div
        draggable={!isGhost}
        onDragStart={(e) => {
           if (isGhost) return;
           const rect = (e.target as HTMLElement).getBoundingClientRect();
           const clientX = e.clientX;
           const clientY = e.clientY;
           const relX = Math.floor((clientX - rect.left) / (CELL_PIXEL_SIZE * scale));
           const relY = Math.floor((clientY - rect.top) / (CELL_PIXEL_SIZE * scale));
           const validX = Math.max(0, Math.min(relX, piece.width - 1));
           const validY = Math.max(0, Math.min(relY, piece.height - 1));
           
           handleDragStart(e, piece, isInventory ? validX : validX, isInventory ? validY : validY);
        }}
        className={`relative ${!isGhost ? 'cursor-grab active:cursor-grabbing hover:scale-105' : ''} transition-transform origin-top-left`}
        style={{
          width: width,
          height: height,
          margin: isInventory ? '12px' : '0',
          opacity: isGhost ? 0.4 : 1,
          pointerEvents: isGhost ? 'none' : 'auto',
          zIndex: isGhost ? 5 : 10
        }}
      >
        {piece.squares.map((sq, idx) => (
          <div
            key={idx}
            className={`absolute border ${isGhost ? 'border-white/50' : 'border-slate-900/50'} flex items-center justify-center font-mono font-bold text-slate-900 select-none pointer-events-none`}
            style={{
              left: sq.x * CELL_PIXEL_SIZE * scale,
              top: sq.y * CELL_PIXEL_SIZE * scale,
              width: sq.size * CELL_PIXEL_SIZE * scale,
              height: sq.size * CELL_PIXEL_SIZE * scale,
              backgroundColor: isGhost ? '#fbbf24' : piece.color, // Ghost is always amber-ish or same color
              boxShadow: isGhost ? 'none' : 'inset 0 0 10px rgba(0,0,0,0.1)',
              fontSize: `${Math.max(12, sq.size * 14 * scale)}px`
            }}
          >
            {sq.size}
          </div>
        ))}
      </div>
    );
  };

  if (!isClient) return <div className="text-white">Loading...</div>;

  const gridCells = Array(GRID_SIZE * GRID_SIZE).fill(null);

  return (
    <div className="min-h-screen bg-slate-900 flex flex-col items-center justify-start p-4 font-sans select-none overflow-auto relative">
      
      {/* Timer (Top Left) */}
      <div className="absolute top-4 left-4 bg-slate-800 text-amber-500 px-4 py-2 rounded-lg border border-slate-700 font-mono text-xl font-bold shadow-lg z-50">
         ⏱ {Math.floor(timeElapsed / 60).toString().padStart(2, '0')}:{ (timeElapsed % 60).toString().padStart(2, '0') }
      </div>

      {/* Header */}
      <div className="mb-4 mt-2 text-center">
        <h1 className="text-3xl font-bold text-white mb-2 tracking-widest uppercase" style={{ textShadow: '0 0 10px #fbbf24' }}>
          完美重叠
        </h1>
      </div>

      {/* Main Game Area */}
      <div className="flex flex-col md:flex-row gap-6 items-start justify-center w-full max-w-[1400px]">
        
        {/* Target Area (The Board) */}
        <div 
          ref={boardRef}
          className="relative bg-slate-800/50 rounded-lg p-2 border-2 border-slate-700 shadow-2xl shrink-0 mx-auto"
          style={{
             width: GRID_SIZE * CELL_PIXEL_SIZE + 20,
             height: GRID_SIZE * CELL_PIXEL_SIZE + 20
          }}
          onDragOver={handleDragOver}
          onDrop={handleDropOnGrid}
          onDragLeave={handleDragLeave}
        >
          <div 
            className="grid relative"
            style={{
              gridTemplateColumns: `repeat(${GRID_SIZE}, ${CELL_PIXEL_SIZE}px)`,
              gridTemplateRows: `repeat(${GRID_SIZE}, ${CELL_PIXEL_SIZE}px)`,
            }}
          >
             {/* Background Cells */}
             {gridCells.map((_, i) => (
               <div key={i} className="w-full h-full border border-slate-700/20"></div>
             ))}

             {/* Ghost Piece (Magnetic Snap Preview) */}
             {ghostPos && draggingPieceId !== null && (
                 <div
                    className="absolute z-20 transition-all duration-75 ease-out"
                    style={{
                        left: ghostPos.x * CELL_PIXEL_SIZE,
                        top: ghostPos.y * CELL_PIXEL_SIZE,
                    }}
                 >
                     {renderPiece(pieces.find(p => p.id === draggingPieceId)!, false, true)}
                 </div>
             )}

             {/* Placed Pieces */}
             {pieces.filter(p => p.position !== null).map(piece => (
               <div
                 key={piece.id}
                 draggable
                 onDragStart={(e) => {
                   const rect = (e.target as HTMLElement).getBoundingClientRect();
                   const relX = Math.floor((e.clientX - rect.left) / CELL_PIXEL_SIZE);
                   const relY = Math.floor((e.clientY - rect.top) / CELL_PIXEL_SIZE);
                   handleDragStart(e, piece, relX, relY);
                 }}
                 className="absolute cursor-grab active:cursor-grabbing z-10 hover:brightness-110 transition-all"
                 style={{
                   left: piece.position!.x * CELL_PIXEL_SIZE,
                   top: piece.position!.y * CELL_PIXEL_SIZE,
                   width: piece.width * CELL_PIXEL_SIZE,
                   height: piece.height * CELL_PIXEL_SIZE,
                 }}
               >
                 {piece.squares.map((sq, idx) => (
                   <div
                     key={idx}
                     className="absolute border border-slate-900/50 flex items-center justify-center font-mono font-bold text-slate-900"
                     style={{
                       left: sq.x * CELL_PIXEL_SIZE,
                       top: sq.y * CELL_PIXEL_SIZE,
                       width: sq.size * CELL_PIXEL_SIZE,
                       height: sq.size * CELL_PIXEL_SIZE,
                       backgroundColor: piece.color,
                       boxShadow: 'inset 0 0 10px rgba(0,0,0,0.1)',
                       fontSize: `${Math.max(12, sq.size * 14)}px`
                     }}
                   >
                     {sq.size}
                   </div>
                 ))}
               </div>
             ))}
          </div>
          
          <div className="absolute -top-1 -left-1 w-6 h-6 border-t-4 border-l-4 border-amber-500"/>
          <div className="absolute -top-1 -right-1 w-6 h-6 border-t-4 border-r-4 border-amber-500"/>
          <div className="absolute -bottom-1 -left-1 w-6 h-6 border-b-4 border-l-4 border-amber-500"/>
          <div className="absolute -bottom-1 -right-1 w-6 h-6 border-b-4 border-r-4 border-amber-500"/>
        </div>

        {/* Inventory / Supply Area - Enlarged and Flexible */}
        <div 
          className="bg-slate-800/80 rounded-xl p-4 w-full md:flex-1 md:min-w-[500px] min-h-[600px] border border-slate-700 flex flex-col shrink-0"
          onDragOver={handleDragOver}
          onDrop={handleDropOnInventory}
        >
          <div className="flex justify-between items-center mb-4 border-b border-slate-600 pb-2">
            <h2 className="text-white font-bold uppercase text-sm tracking-wider">
               Fragments ({pieces.filter(p => p.position === null).length})
            </h2>
            <button 
                onClick={handleHint}
                className={`px-3 py-1 bg-amber-600 hover:bg-amber-500 text-white text-xs rounded uppercase font-bold tracking-wide transition-colors shadow-lg ${!isGameActive ? 'opacity-50 cursor-not-allowed' : ''}`}
                title="Automatically place one piece"
                disabled={!isGameActive}
            >
                Hint 💡
            </button>
          </div>
          
          <div className="flex flex-wrap gap-2 justify-start items-start flex-1 content-start">
            {pieces.filter(p => p.position === null).map(piece => (
              <div key={piece.id} className="transition-all hover:-translate-y-1">
                {renderPiece(piece, true)}
              </div>
            ))}
          </div>
        </div>

      </div>
      
      {/* Footer */}
      <div className="mt-8 flex gap-4 relative z-30">
        <button 
          onClick={startNewGame}
          className="px-8 py-3 bg-slate-700 hover:bg-slate-600 text-white rounded font-bold uppercase tracking-wide transition-colors border-2 border-slate-500 cursor-pointer"
        >
          New Challenge
        </button>

        <button 
          type="button"
          onClick={handleSubmit}
          className="px-8 py-3 bg-emerald-700 hover:bg-emerald-600 text-white rounded font-bold uppercase tracking-wide transition-colors border-2 border-emerald-500 shadow-lg cursor-pointer active:scale-95 transform"
        >
          提交判定
        </button>
      </div>

      {/* Modal */}
      {modal && modal.show && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-in fade-in duration-200">
            <div className={`bg-slate-800 border-2 ${modal.type === 'success' ? 'border-amber-500' : 'border-slate-600'} rounded-xl p-8 max-w-md w-full shadow-2xl transform transition-all scale-100`}>
                <h3 className={`text-2xl font-bold mb-4 ${modal.type === 'success' ? 'text-amber-500' : 'text-white'}`}>{modal.title}</h3>
                <p className="text-slate-300 text-lg mb-8 whitespace-pre-line">{modal.message}</p>
                <button 
                    onClick={() => setModal(null)}
                    className="w-full py-3 bg-slate-700 hover:bg-slate-600 text-white rounded font-bold uppercase tracking-wide transition-colors border border-slate-500"
                >
                    关闭
                </button>
            </div>
        </div>
      )}

    </div>
  );
};

const root = createRoot(document.getElementById('app')!);
root.render(<Game />);
