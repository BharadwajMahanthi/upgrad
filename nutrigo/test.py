import unittest
from app import app
import json

class AppTestCase(unittest.TestCase):
    def setUp(self):
        # Set up the test client
        app.testing = True
        self.app = app.test_client()
    
    def test_home_page(self):
        # Test that the home page loads correctly
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Welcome to the Recipe Recommendation System', response.data)
    
    def test_recommend_valid_user(self):
        # Test the recommend route with a valid user ID
        response = self.app.post('/recommend', data=dict(
            user_id='123',  # Replace with a valid user ID from your dataset
            preferences=''
        ), follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Your Recipe Recommendations', response.data)
        # You can add more assertions to check the content of the recommendations
    
    def test_recommend_invalid_user(self):
        # Test the recommend route with an invalid user ID
        response = self.app.post('/recommend', data=dict(
            user_id='invalid_user',
            preferences=''
        ), follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'No recommendations found', response.data)
    
    def test_recommend_with_preferences(self):
        # Test the recommend route with preferences
        response = self.app.post('/recommend', data=dict(
            user_id='123',  # Replace with a valid user ID
            preferences='vegan, gluten-free'
        ), follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Your Recipe Recommendations', response.data)
        # You can add more assertions to verify that the recommendations match the preferences

if __name__ == '__main__':
    unittest.main()
