# Student Photos

Add student photos here to register them for face recognition.

## Structure

Create one subfolder per student, named with the student's name:

```
student_photos/
├── John_Doe/
│   ├── img_0.jpg
│   ├── img_1.jpg
│   └── img_2.jpg
├── Jane_Smith/
│   └── img_0.jpg
```

## Guidelines

- Use 3–5 photos per student for best accuracy
- Photos should clearly show the face (good lighting, facing forward)
- Supported formats: .jpg, .jpeg, .png
- After adding photos, run /train (via mobile app admin panel or curl)

## Notes

- Folder name = student ID used in attendance records
- Use only alphanumeric characters, spaces, hyphens, underscores in folder names
- You can also add students via the mobile app's "Add Student" feature
