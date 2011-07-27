package org.graphstream.util;

public class NativePointer {
    private long __32bits;
    private long __64bits;

    public NativePointer() {
	__32bits = 0;
	__64bits = 0;
    }

    public native long getPointer();
    public native void setPointer(long pointer);
}
