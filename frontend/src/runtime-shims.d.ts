declare module 'react' {
  export type ChangeEvent<T = Element> = { target: HTMLInputElement }
  export function useState<T>(initialValue: T): [T, (value: T) => void]
  export const StrictMode: any
}

declare module 'react-dom/client' {
  export function createRoot(element: Element): { render(node: any): void }
}

declare module 'lucide-react' {
  export const Activity: any
  export const ArrowUpRight: any
  export const CheckCircle2: any
  export const FileImage: any
  export const ScanSearch: any
  export const ShieldCheck: any
  export const Upload: any
  export const Zap: any
}

declare module 'react/jsx-runtime' {
  export const jsx: any
  export const jsxs: any
  export const Fragment: any
}

declare namespace JSX {
  interface IntrinsicElements {
    [elementName: string]: any
  }
}

declare module '*.css'
